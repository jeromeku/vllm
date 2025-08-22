// vmm_allocator.cpp
// Build: g++ -std=c++17 -fPIC -shared -O2 \
//   -I${CUDA_HOME:-/usr/local/cuda}/include \
//   vmm_allocator.cpp -o libvmm_alloc.so \
//   -L${CUDA_HOME:-/usr/local/cuda}/lib64 -lcuda -ldl
//
// Minimal CUDA VMM-backed allocator implementing PyTorch's CUDAPluggableAllocator ABI.
// Exports vmm_malloc / vmm_free (ABI) and vmm_sleep / vmm_wake / vmm_num_tracked.
//
// Semantics:
//  - vmm_malloc: reserve VA (first time), create physical allocation, map, set RW access.
//  - vmm_free:   unmap + release handle, free VA, drop tracking.
//  - vmm_sleep:  for all tracked segments: unmap + release handle, KEEP VA reserved.
//  - vmm_wake:   for all tracked segments that are "asleep": recreate handle + remap same VA.
//  - Pointer values are stable across sleep/wake; touching memory while asleep is UB.
//
// Threading:
//  - g_mu protects the segment table ONLY.
//  - g_ctx_mu protects the per-device primary-context cache ONLY.
//  - No driver calls are made while holding g_mu (prevents deadlocks).

#include <cuda.h>
#include <cuda_runtime_api.h>  // for cudaStream_t type only
#include <unordered_map>
#include <vector>
#include <mutex>
#include <cstdio>
#include <cinttypes>
#include <pybind11/pybind11.h>

#define CHECK_CUDA_DRV(expr) do {                                      \
  CUresult _res = (expr);                                              \
  if (_res != CUDA_SUCCESS) {                                          \
    const char* _name = nullptr;                                       \
    const char* _str  = nullptr;                                       \
    cuGetErrorName(_res, &_name);                                      \
    cuGetErrorString(_res, &_str);                                     \
    std::fprintf(stderr, "CUDA-DRV error %s: %s (%d) at %s:%d\n",      \
      _name ? _name : "?", _str ? _str : "?", (int)_res,               \
      __FILE__, __LINE__);                                             \
    std::abort();                                                      \
  }                                                                    \
} while (0)

namespace {

// Tracked segment state.
struct Segment {
  CUdeviceptr addr{0};                      // reserved virtual address (VA)
  size_t      size{0};                      // bytes (granularity-aligned)
  CUmemGenericAllocationHandle handle{0};   // physical allocation handle (0 when asleep)
  int         device{0};                    // CUDA device id
  bool        mapped{false};                // whether VA currently has a mapping
};

// --- Global state

// All live segments keyed by VA (as void* for convenience).
static std::unordered_map<void*, Segment> g_segments;
static std::mutex g_mu;  // protects g_segments

// Per-device primary context cache (matches PyTorch usage of primary ctx).
static std::unordered_map<int, CUcontext> g_device_ctx;
static std::mutex g_ctx_mu;  // protects g_device_ctx

// --- Helpers

static inline size_t align_up(size_t n, size_t a) {
  return (n + a - 1) / a * a;
}

// Ensure the device's primary context is current in this thread.
// Uses a separate mutex (g_ctx_mu) so it never contends with g_mu.
static void ensure_device_ctx(int device) {
  std::lock_guard<std::mutex> lock(g_ctx_mu);
  auto it = g_device_ctx.find(device);
  if (it != g_device_ctx.end()) {
    CHECK_CUDA_DRV(cuCtxSetCurrent(it->second));
    return;
  }
  CUdevice dev;
  CHECK_CUDA_DRV(cuDeviceGet(&dev, device));
  CUcontext ctx;
  CHECK_CUDA_DRV(cuDevicePrimaryCtxRetain(&ctx, dev));
  CHECK_CUDA_DRV(cuCtxSetCurrent(ctx));
  g_device_ctx.emplace(device, ctx);
}

// Create physical allocation + map it into seg.addr, set RW access.
// Assumes seg.size is requested size (will be aligned), and seg.addr may be 0 (first map).
static void create_and_map(Segment& seg) {
  ensure_device_ctx(seg.device);

  CUmemAllocationProp prop{};
  prop.type                         = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type                = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id                  = seg.device;
  prop.requestedHandleTypes         = CU_MEM_HANDLE_TYPE_NONE;

  size_t gran_min = 0;
  CHECK_CUDA_DRV(cuMemGetAllocationGranularity(&gran_min, &prop,
                                               CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  seg.size = align_up(seg.size, gran_min);

  if (seg.addr == 0) {
    // Let the driver choose an aligned VA range.
    CHECK_CUDA_DRV(cuMemAddressReserve(&seg.addr, seg.size, gran_min, 0, 0));
  }

  CHECK_CUDA_DRV(cuMemCreate(&seg.handle, seg.size, &prop, 0));
  CHECK_CUDA_DRV(cuMemMap(seg.addr, seg.size, 0, seg.handle, 0));

  CUmemAccessDesc access{};
  access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  access.location.id   = seg.device;
  access.flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CHECK_CUDA_DRV(cuMemSetAccess(seg.addr, seg.size, &access, 1));

  seg.mapped = true;
}

// Unmap current mapping (if any) and release the physical handle.
// Keeps VA reserved. Assumes correct device ctx is current.
static void unmap_and_release(Segment& seg) {
  if (!seg.addr || !seg.mapped) return;
  CHECK_CUDA_DRV(cuMemUnmap(seg.addr, seg.size));
  if (seg.handle) {
    CHECK_CUDA_DRV(cuMemRelease(seg.handle));
    seg.handle = 0;
  }
  seg.mapped = false;
}

} // namespace

extern "C" {

// Make sure the Driver API is initialized when the .so is loaded.
__attribute__((constructor))
static void vmm_ctor() {
  CHECK_CUDA_DRV(cuInit(0));
}

// ------------------------------
// PyTorch CUDAPluggableAllocator
// ------------------------------

// Allocate: create + map a VMM segment on 'device' and return the VA pointer.
__attribute__((visibility("default")))
void* vmm_malloc(ssize_t size, int device, cudaStream_t /*stream*/) {
  if (size <= 0) size = 1;  // be robust to zero-size
  Segment seg{};
  seg.device = device;
  seg.size   = static_cast<size_t>(size);

  // Create + map (no g_mu held during driver calls).
  create_and_map(seg);

  void* ptr = reinterpret_cast<void*>(seg.addr);

  // Record under lock.
  {
    std::lock_guard<std::mutex> lk(g_mu);
    g_segments[ptr] = seg;
  }
  return ptr;
}

// Free: unmap + release + free VA. Removes tracking entry.
__attribute__((visibility("default")))
void vmm_free(void* ptr, size_t /*size*/, int device, cudaStream_t /*stream*/) {
  if (!ptr) return;

  // Fetch and erase the entry under lock.
  Segment seg{};
  {
    std::lock_guard<std::mutex> lk(g_mu);
    auto it = g_segments.find(ptr);
    if (it == g_segments.end()) return;  // idempotent
    seg = it->second;
    g_segments.erase(it);
  }

  // Make sure we're operating on the right device.
  ensure_device_ctx(seg.device);

  // Tear down mapping + handle, then free VA.
  if (seg.mapped) {
    CHECK_CUDA_DRV(cuMemUnmap(seg.addr, seg.size));
    if (seg.handle) CHECK_CUDA_DRV(cuMemRelease(seg.handle));
  }
  CHECK_CUDA_DRV(cuMemAddressFree(seg.addr, seg.size));
}

// ---------------------------------
// Bulk sleep/wake/introspection APIs
// ---------------------------------

// Unmap + release all physical allocations but keep their VAs reserved.
__attribute__((visibility("default")))
void vmm_sleep() {
  // Snapshot keys first to avoid holding g_mu during driver calls.
  std::vector<void*> keys;
  {
    std::lock_guard<std::mutex> lk(g_mu);
    keys.reserve(g_segments.size());
    for (auto& kv : g_segments) keys.push_back(kv.first);
  }

  for (void* key : keys) {
    // Copy current state under lock.
    Segment seg{};
    {
      std::lock_guard<std::mutex> lk(g_mu);
      auto it = g_segments.find(key);
      if (it == g_segments.end()) continue;
      seg = it->second;
    }

    // Driver operations outside g_mu.
    ensure_device_ctx(seg.device);
    unmap_and_release(seg);  // modifies local 'seg' only

    // Write back mapped/handle state under lock.
    {
      std::lock_guard<std::mutex> lk(g_mu);
      auto it = g_segments.find(key);
      if (it != g_segments.end()) {
        it->second.mapped = false;
        it->second.handle = 0;
      }
    }
  }
}

// Recreate physical allocations and remap them to the same VA.
__attribute__((visibility("default")))
void vmm_wake() {
  // Snapshot keys.
  std::vector<void*> keys;
  {
    std::lock_guard<std::mutex> lk(g_mu);
    keys.reserve(g_segments.size());
    for (auto& kv : g_segments) keys.push_back(kv.first);
  }

  for (void* key : keys) {
    // Copy under lock.
    Segment seg{};
    {
      std::lock_guard<std::mutex> lk(g_mu);
      auto it = g_segments.find(key);
      if (it == g_segments.end()) continue;
      seg = it->second;
    }

    if (!seg.mapped) {
      ensure_device_ctx(seg.device);
      create_and_map(seg);  // seg.mapped=true and seg.handle set

      // Write back refreshed state.
      std::lock_guard<std::mutex> lk(g_mu);
      auto it = g_segments.find(key);
      if (it != g_segments.end()) it->second = seg;
    }
  }
}

// Number of tracked segments (for debugging).
__attribute__((visibility("default")))
size_t vmm_num_tracked() {
  std::lock_guard<std::mutex> lk(g_mu);
  return g_segments.size();
}

// -------------------------
// pybind11 module
// -------------------------
PYBIND11_MODULE(_vmm_alloc, m) {
  m.doc() = "CUDA VMM allocator with sleep/wake, exportable to PyTorch MemPool";

  m.def("sleep", &vmm_sleep, "Unmap+release all physical allocations (keep VA).");
  m.def("wake",  &vmm_wake,  "Recreate physical allocations and remap to same VAs.");
  m.def("num_tracked", []() { return static_cast<uint64_t>(vmm_num_tracked()); },
        "Number of live tracked segments.");
  // Expose addresses for debugging (optional).
  m.def("_tracked_ptrs", []() {
    std::lock_guard<std::mutex> lk(g_mu);
    std::vector<uint64_t> addrs;
    addrs.reserve(g_segments.size());
    for (auto& kv : g_segments) addrs.push_back(reinterpret_cast<uint64_t>(kv.first));
    return addrs;
  });

  // Expose the symbol names expected by CUDAPluggableAllocator (documentation aid)
  m.attr("ALLOC_SYMBOL") = "vmm_malloc";
  m.attr("FREE_SYMBOL")  = "vmm_free";
}  
} // extern "C"

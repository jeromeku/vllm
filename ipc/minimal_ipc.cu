// cuda_ipc_kernel_ext.cu
// Minimal CUDA IPC + kernels that directly dereference peer memory.
// - Supports CUDA 12+ via driver API cuMemGetAddressRange
// - Handles PyTorch suballocations (exports handle for base + byte offset)
// - Returns both base and offset pointers so you can close the right one.

#include <torch/extension.h>

#include <pybind11/pybind11.h>
#include <cuda_runtime.h>
#include <cuda.h>

#include <cstring>
#include <stdexcept>
#include <string>

namespace py = pybind11;
using namespace pybind11::literals;

// ---------- error helpers ----------
static inline void check_cuda(cudaError_t st, const char* where = "") {
  if (st != cudaSuccess) {
    std::string msg = where;
    if (!msg.empty()) msg += ": ";
    msg += cudaGetErrorString(st);
    throw std::runtime_error(msg);
  }
}

static inline void check_drv(CUresult st, const char* where = "") {
  if (st != CUDA_SUCCESS) {
    const char* err = nullptr;
    cuGetErrorString(st, &err);
    std::string msg = where;
    if (!msg.empty()) msg += ": ";
    msg += (err ? err : "CUDA driver error");
    throw std::runtime_error(msg);
  }
}

// ---------- kernels (float32 demo) ----------
__global__ void k_copy_from_remote(float* __restrict__ dst,
                                   const float* __restrict__ src,
                                   size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = src[i];
}

__global__ void k_add_inplace_remote(float* __restrict__ p,
                                     float val,
                                     size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] += val;
}

// ---------- allocation base + span ----------
static inline void get_allocation_base_and_span(void* ptr, void** out_base, size_t* out_span) {
#if CUDART_VERSION >= 12000
  // CUDA 12+: runtime removed cudaMemGetAddressRange; use driver API
  check_drv(cuInit(0), "cuInit");
  CUdeviceptr base = 0;
  size_t span = 0;
  check_drv(cuMemGetAddressRange(&base, &span, reinterpret_cast<CUdeviceptr>(ptr)),
            "cuMemGetAddressRange");
  *out_base = reinterpret_cast<void*>(base);
  *out_span = span;
#else
  void* base = nullptr;
  size_t span = 0;
  check_cuda(cudaMemGetAddressRange(&base, &span, ptr), "cudaMemGetAddressRange");
  *out_base = base;
  *out_span = span;
#endif
}

// ---------- IPC API exposed to Python ----------

// Return (handle_bytes, byte_offset) for a contiguous float32 CUDA tensor.
py::tuple export_ipc_handle_and_offset(torch::Tensor t) {
  TORCH_CHECK(t.is_cuda(), "tensor must be on CUDA");
  TORCH_CHECK(t.is_contiguous(), "tensor must be contiguous");
  TORCH_CHECK(t.scalar_type() == at::kFloat, "demo binds float32 only");

  void* base = nullptr;
  size_t span = 0;
  get_allocation_base_and_span(t.data_ptr(), &base, &span);

  // Byte offset of the tensor start within the underlying allocation.
  const size_t offset = static_cast<const char*>(t.data_ptr()) - static_cast<char*>(base);

  cudaIpcMemHandle_t h{};
  // IMPORTANT: handle must be for the BASE allocation.
  check_cuda(cudaIpcGetMemHandle(&h, base), "cudaIpcGetMemHandle");

  py::bytes hb(reinterpret_cast<const char*>(&h), sizeof(h));
  return py::make_tuple(hb, static_cast<uint64_t>(offset));
}

// Open a remote allocation from a handle; returns BASE pointer in receiver VA space.
uintptr_t open_remote_base(py::bytes handle_bytes) {
  std::string buf = handle_bytes;
  TORCH_CHECK(buf.size() == sizeof(cudaIpcMemHandle_t), "bad IPC handle size");
  cudaIpcMemHandle_t h{};
  std::memcpy(&h, buf.data(), sizeof(h));

  void* remote_base = nullptr;
  check_cuda(cudaIpcOpenMemHandle(&remote_base, h, cudaIpcMemLazyEnablePeerAccess),
             "cudaIpcOpenMemHandle");
  return reinterpret_cast<uintptr_t>(remote_base);
}

// Add a byte offset to a BASE pointer (pure arithmetic; does not open/close anything).
uintptr_t add_offset(uintptr_t base_ptr, uint64_t byte_offset) {
  return static_cast<uintptr_t>(base_ptr + byte_offset);
}

// Close a previously opened BASE pointer (must pass the BASE returned by open_remote_base).
void close_remote_base(uintptr_t remote_base_ptr) {
  check_cuda(cudaIpcCloseMemHandle(reinterpret_cast<void*>(remote_base_ptr)),
             "cudaIpcCloseMemHandle");
}

// Enable P2P if possible (no-op if unsupported or already enabled).
void enable_peer_access(int from_dev, int to_dev) {
  int can = 0;
  check_cuda(cudaDeviceCanAccessPeer(&can, from_dev, to_dev), "cudaDeviceCanAccessPeer");
  if (can) {
    check_cuda(cudaSetDevice(from_dev), "cudaSetDevice(from_dev)");
    // Ignore "already enabled" by clearing last error after try
    cudaError_t s = cudaDeviceEnablePeerAccess(to_dev, 0);
    if (s != cudaSuccess && s != cudaErrorPeerAccessAlreadyEnabled) {
      check_cuda(s, "cudaDeviceEnablePeerAccess");
    }
    (void)cudaGetLastError(); // clear sticky
  }
}

// Launch a kernel on dst.device() that reads from remote_ptr and writes into dst (n_elems float32).
void copy_from_remote(torch::Tensor dst, uintptr_t remote_ptr, size_t n_elems) {
  TORCH_CHECK(dst.is_cuda() && dst.is_contiguous(), "dst must be contiguous CUDA tensor");
  TORCH_CHECK(dst.scalar_type() == at::kFloat, "demo binds float32 only");
  TORCH_CHECK(static_cast<size_t>(dst.numel()) == n_elems, "n_elems mismatch");

  const int dev = dst.get_device();
  check_cuda(cudaSetDevice(dev), "cudaSetDevice(dst.device)");

  dim3 blk(256), grd((n_elems + blk.x - 1) / blk.x);
  k_copy_from_remote<<<grd, blk>>>(dst.data_ptr<float>(),
                                   reinterpret_cast<const float*>(remote_ptr),
                                   n_elems);
  check_cuda(cudaGetLastError(), "k_copy_from_remote launch");
  check_cuda(cudaDeviceSynchronize(), "copy_from_remote sync");
}

// Launch a kernel on on_device that adds 'val' into remote_ptr in-place (n_elems float32).
void add_inplace_remote(uintptr_t remote_ptr, float val, size_t n_elems, int on_device) {
  check_cuda(cudaSetDevice(on_device), "cudaSetDevice(on_device)");
  dim3 blk(256), grd((n_elems + blk.x - 1) / blk.x);
  k_add_inplace_remote<<<grd, blk>>>(reinterpret_cast<float*>(remote_ptr), val, n_elems);
  check_cuda(cudaGetLastError(), "k_add_inplace_remote launch");
  check_cuda(cudaDeviceSynchronize(), "add_inplace_remote sync");
}

// ---------- pybind11 module ----------
PYBIND11_MODULE(cuda_ipc_kernel_ext, m) {
  m.doc() = "CUDA IPC + kernel dereference demo (float32) with handle+offset support";

  m.def("export_ipc_handle_and_offset", &export_ipc_handle_and_offset,
        "t"_a,
        "Export (IPC handle for base allocation, byte offset) for a contiguous float32 CUDA tensor.");

  m.def("open_remote_base", &open_remote_base,
        "handle_bytes"_a,
        "Open a remote allocation from an IPC handle; returns BASE pointer.");

  m.def("add_offset", &add_offset,
        "base_ptr"_a, "byte_offset"_a,
        "Add byte offset to BASE pointer to get the tensor's device address.");

  m.def("close_remote_base", &close_remote_base,
        "base_ptr"_a,
        "Close a previously opened BASE pointer.");

  m.def("enable_peer_access", &enable_peer_access,
        "from_dev"_a, "to_dev"_a,
        "Enable P2P access if available (best-effort).");

  m.def("copy_from_remote", &copy_from_remote,
        "dst"_a, "remote_ptr"_a, "n_elems"_a,
        "Kernel: copy n_elems float32 from remote_ptr -> dst (on dst.device())");

  m.def("add_inplace_remote", &add_inplace_remote,
        "remote_ptr"_a, "val"_a, "n_elems"_a, "on_device"_a,
        "Kernel: add val into remote_ptr in-place (n_elems float32) on on_device");
}

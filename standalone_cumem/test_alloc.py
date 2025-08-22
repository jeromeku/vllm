# test_remap.py
import ctypes, os, torch
from torch.cuda import memory as tcm

so_path = os.path.abspath("./libvmm_alloc.so")
lib = ctypes.CDLL(so_path)
KB = 2**10
MB = KB * KB
GB = MB * KB


def print_mem_info(tag: str):
    free, total = [m / 1e9 for m in torch.cuda.mem_get_info()]
    mem_alloc = torch.cuda.memory_allocated() / 1e9
    mem_reserved = torch.cuda.memory_reserved() / 1e9
    print(f"{tag}: driver {free:.2f} / {total:.2f} | torch alloc={mem_alloc:.2f} res={mem_reserved:.2f}")


# function prototypes (optional, for type safety in ctypes)
lib.vmm_sleep.restype = None
lib.vmm_wake.restype = None
lib.vmm_num_tracked.restype = ctypes.c_size_t

# 1) Load allocator and switch MemPool routing to it
pluggable = tcm.CUDAPluggableAllocator(so_path, "vmm_malloc", "vmm_free")  # ABI per docs
pool = tcm.MemPool(pluggable.allocator())  # route this pool to our allocator

torch.cuda.synchronize()
free0, total = torch.cuda.mem_get_info()
print_mem_info("BEFORE ALLOC")

# print(
#     f"Before: free={free0 / 1e9:.2f} GB total={total / 1e9:.2f} GB | torch {alloc=:.2f} {reserved=:.2f}"
# )

# 2) Allocate "weights" inside the pool
with tcm.use_mem_pool(pool):
    w = torch.ones((10 * GB,), device="cuda", dtype=torch.uint8)
    # w.fill_(3.14)
    ptr_before = w.data_ptr()

torch.cuda.synchronize()
free1, _ = torch.cuda.mem_get_info()
# alloc, reserved = get_torch_cuda_mem()
# print(
#     f"After allocate: free={free1 / 1e9:.2f} GB, tracked={lib.vmm_num_tracked()} ptr=0x{ptr_before:x} | torch {alloc=:.2f} {reserved:.2f}"
# )
print_mem_info("AFTER ALLOC")

# 3) Sleep (unmap+release physical memory; VA remains reserved)
lib.vmm_sleep()
torch.cuda.synchronize()
print_mem_info("AFTER SLEEP")

# 4) Do other work with default allocator (outside the pool)
tmp = torch.ones((5 * GB,), device="cuda", dtype=torch.uint8)  # just to show memory can be reused
print_mem_info("AFTER DEFAULT POOL")
del tmp
torch.cuda.empty_cache()
torch.cuda.synchronize()
print_mem_info("AFTER DEL FROM DEFAULT POOL")

# 5) Wake (recreate physical memory backing the same VA)
lib.vmm_wake()
torch.cuda.synchronize()
print_mem_info("AFTER WAKE")

# 6) Verify the pointer is identical and we can write again
ptr_after = w.data_ptr()
print(f"Pointer stable? {ptr_before == ptr_after} (before=0x{ptr_before:x}, after=0x{ptr_after:x})")

# Touch memory (this would have crashed if we hadn’t remapped)
w.zero_()
print("w.zero_() succeeded post-wake; contents reset (as expected).")

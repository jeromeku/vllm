import os
from torch.utils.cpp_extension import load
import subprocess
import shutil

BUILD_DIR = "./build"

# shutil.rmtree(BUILD_DIR, ignore_errors=True)
os.makedirs(BUILD_DIR, exist_ok=True)
LIBNAME = "cuda_ipc_kernel_ext"
SONAME = f"{LIBNAME}.so"
SOURCES = ["minimal_ipc.cu"]
od = load(
    name=LIBNAME,
    sources=SOURCES,
    extra_cuda_cflags=["-O3", "-std=c++17"],
    extra_ldflags=["-lcuda"],   # driver lib for cuMemGetAddressRange on CUDA 12+
    verbose=True,
    build_directory=BUILD_DIR
)

CUR_DIR = os.path.dirname(__file__)
print(f"{CUR_DIR}")

os.symlink(f"{BUILD_DIR}/{SONAME}", f"{CUR_DIR}/{SONAME}")

print("Built:", f"{CUR_DIR}/{SONAME}")

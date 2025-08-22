# 0) Prereqs:
#    - Linux, CUDA 12.x (or 11.x with VMM), PyTorch >= 2.2 (tested path); a GPU that supports VMM.
#    - Make sure /usr/local/cuda points to your CUDA toolkit.

# 1) Build the .so
g++ -std=c++17 -fPIC -shared -O2 -I/usr/local/cuda/include \
  vmm_allocator.cpp -o libvmm_alloc.so \
  -L/usr/local/cuda/lib64 -lcuda -ldl

# 2) Install torch if needed, then run:
python3 test_remap.py

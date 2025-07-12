// Build commands
// ptx
// nvcc --ptx --gpu-architecture=compute_80 vector_add.cu -o vector_add.ptx
// cubin
// nvcc --cubin --gpu-architecture=compute_80 --gpu-code=sm_86 vector_add.cu -o
// vector_add.cubin
// fatbin
// nvcc --fatbin --gpu-architecture=compute_80 --gpu-code=sm_86 vector_add.cu -o
// vector_add.fatbin

#include <cuda_runtime.h>

// The extern "C" is necessary.
extern "C" __global__ void vector_add(int const* a, int const* b, int* c, unsigned int n)
{
    unsigned int const stride{blockDim.x * gridDim.x};
    unsigned int const start_idx{blockDim.x * blockIdx.x + threadIdx.x};
    for (unsigned int i{start_idx}; i < n; i += stride)
    {
        c[i] = a[i] + b[i];
    }
}
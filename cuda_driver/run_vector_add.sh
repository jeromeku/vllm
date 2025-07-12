#!/bin/bash

set -euo pipefail

PROGRAM="loadPtx.cu"
EMBEDDED_PTX="/home/jeromeku/vllm/cuda_driver/binaries/vector_add.cubin.cpp"
CUDA_ROOT="/home/jeromeku/kernels/cuda"

CMD="nvcc ${PROGRAM} ${EMBEDDED_PTX} -I${CUDA_ROOT}/include -L${CUDA_ROOT}/lib64 -lcuda -o vec_add"
echo $CMD
eval $CMD

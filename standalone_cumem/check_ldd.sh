#!/bin/bash

LIB_PATH="./build/libvmm_alloc.so"

CMD="readelf -d ${LIB_PATH} | grep -E 'RPATH|RUNPATH'"
echo "${CMD}"
eval "${CMD}"

echo -E "\n"


CMD="ldd ${LIB_PATH}"
echo "${CMD}"
eval "${CMD}"
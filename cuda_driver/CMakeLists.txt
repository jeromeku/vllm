cmake_minimum_required(VERSION 3.28)

project(Build-CUBIN VERSION 0.0.1 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA Toolkit
find_package(CUDAToolkit REQUIRED)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/generated")
message(STATUS ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

# PTX
add_library(vector_add_ptx OBJECT vector_add.cu)
set_target_properties(vector_add_ptx PROPERTIES CUDA_PTX_COMPILATION ON)
set_target_properties(vector_add_ptx PROPERTIES CUDA_ARCHITECTURES "90-virtual")
message(STATUS ${CMAKE_BINARY_DIR})

# # CUBIN
add_library(vector_add_cubin OBJECT vector_add.cu)
set_target_properties(vector_add_cubin PROPERTIES CUDA_CUBIN_COMPILATION ON)
set_target_properties(vector_add_cubin PROPERTIES CUDA_ARCHITECTURES "90-real")

add_library(vector_add_fatbin OBJECT vector_add.cu)
set_target_properties(vector_add_fatbin PROPERTIES CUDA_FATBIN_COMPILATION ON)
set_target_properties(vector_add_fatbin PROPERTIES CUDA_ARCHITECTURES "90-real;90-virtual")

# add_library(vector_add_cubin_sm70 OBJECT vector_add.cu)
# set_target_properties(vector_add_cubin_sm70 PROPERTIES CUDA_CUBIN_COMPILATION ON)
# set_target_properties(vector_add_cubin_sm70 PROPERTIES CUDA_ARCHITECTURES "70-real")

# # FATBIN
# add_library(vector_add_fatbin OBJECT vector_add.cu)
# set_target_properties(vector_add_fatbin PROPERTIES CUDA_FATBIN_COMPILATION ON)
# set_target_properties(vector_add_fatbin PROPERTIES CUDA_ARCHITECTURES "80-virtual;80-real;80-real")
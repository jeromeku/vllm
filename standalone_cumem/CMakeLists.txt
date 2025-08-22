# CMakeLists.txt
cmake_minimum_required(VERSION 3.18)
project(vmm_alloc CXX)
find_package(CUDAToolkit REQUIRED)

# Collect all IMPORTED targets with the CUDA:: prefix
get_property(_all_targets GLOBAL PROPERTY TARGETS)
set(_cuda_targets)
foreach(t IN LISTS _all_targets)
  if(TARGET ${t})
    get_target_property(_is_imported ${t} IMPORTED)
    if(_is_imported AND t MATCHES "^CUDA::")
      list(APPEND _cuda_targets ${t})
    endif()
  endif()
endforeach()
list(SORT _cuda_targets)
message(STATUS "CUDAToolkit imported targets: ${_cuda_targets}")


# find_library(CUDA_LIB cuda HINTS /usr/local/cuda/lib64)

# get_target_property(_ifaces CUDA::cudart INTERFACE_LINK_LIBRARIES)
# message(STATUS "CUDA::cudart ifaces = ${_ifaces}")


# include_directories(/usr/local/cuda/include)
add_library(vmm_alloc SHARED vmm_allocator.cpp)
# target_link_libraries(vmm_alloc ${CUDA_LIB} dl)

target_link_libraries(vmm_alloc PRIVATE CUDA::cuda_driver ${CMAKE_DL_LIBS})
target_include_directories(vmm_alloc PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
set_target_properties(vmm_alloc PROPERTIES OUTPUT_NAME vmm_alloc PREFIX "lib")
set_target_properties(vmm_alloc PROPERTIES
  BUILD_RPATH  "$ORIGIN"
  INSTALL_RPATH "$ORIGIN")

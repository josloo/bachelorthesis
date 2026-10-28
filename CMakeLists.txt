cmake_minimum_required(VERSION 3.20)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler --expt-relaxed-constexpr --extended-lambda -lineinfo")
project(BA CUDA)
find_package(CUDAToolkit)
set(CMAKE_CUDA_STANDARD 20)

add_subdirectory(benchmarks)

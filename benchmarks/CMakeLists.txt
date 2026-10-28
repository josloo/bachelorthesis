include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/CPM.cmake")

CPMAddPackage(
        URI "gh:NVIDIA/nvbench#main"
        OPTIONS
            "NVBench_ENABLE_CUPTI OFF"
)

add_library(kernels INTERFACE)

target_sources(
    kernels
    PUBLIC
        FILE_SET HEADERS
        BASE_DIRS
            include
        FILES
            #"../archiv/kernels_archive.cuh" alt
            "include/kernels_strided.cuh"
)

function(add_version name)
    add_executable(${name} "${CMAKE_CURRENT_SOURCE_DIR}/src/${name}.cu")
    target_link_libraries(${name} kernels CUDA::cudart nvbench::main)
    set_target_properties(${name} PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON)
endfunction()

add_version(baseline_0)
add_version(baseline_1)

add_version(approach_0)
add_version(approach_1)
add_version(approach_2)
add_version(approach_3)

add_version(scaling)
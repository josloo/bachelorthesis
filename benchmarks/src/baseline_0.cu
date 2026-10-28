#include <nvbench/nvbench.cuh>
#include "kernels_strided.cuh"
#include <cuda/cmath>
#include <numeric>
#include <random>
#include <algorithm>

void bench_baseline_0(nvbench::state& state) {
    const size_t num_elems = state.get_int64("NumElems");
    const size_t iterations_mi = state.get_int64("Iterations_MI");
    const size_t iterations_ci = state.get_int64("Iterations_CI");


    // 1. Host-Array erstellen und mit 0, 1, 2, ..., N-1 füllen
    std::vector<size_t> h_mi_indices(num_elems);
    std::iota(h_mi_indices.begin(), h_mi_indices.end(), 0);

    // 2. Array zufällig mischen (Permutation)
    std::mt19937 g(42); // Fester Seed für Reproduzierbarkeit
    std::shuffle(h_mi_indices.begin(), h_mi_indices.end(), g);


    const size_t elements_ci = num_elems;
    const size_t elements_mi = num_elems;

    const size_t size_ci = elements_ci * sizeof(float);
    const size_t size_mi = elements_mi * sizeof(float);
    const size_t size_mi_indices = elements_mi * sizeof(size_t);

    float* d_ci_data = nullptr;
    float* d_mi_in = nullptr;
    float* d_mi_out = nullptr;
    size_t* d_mi_indices = nullptr;

    cudaMalloc(&d_ci_data, size_ci);
    cudaMalloc(&d_mi_in, size_mi);
    cudaMalloc(&d_mi_out, size_mi);
    cudaMalloc(&d_mi_indices, size_mi_indices);

    cudaMemset(d_ci_data, 0, size_ci);
    cudaMemset(d_mi_in, 0, size_mi);
    cudaMemset(d_mi_out, 0, size_mi);
    cudaMemcpy(d_mi_indices, h_mi_indices.data(), size_mi_indices, cudaMemcpyHostToDevice);


    // Grid zur Laufzeit ableiten statt fester 4090-Konstanten
    int dev = 0;
    cudaGetDevice(&dev);
    int sm_count = 0, max_threads_per_sm = 0;
    cudaDeviceGetAttribute(&sm_count,           cudaDevAttrMultiProcessorCount,         dev);
    cudaDeviceGetAttribute(&max_threads_per_sm, cudaDevAttrMaxThreadsPerMultiProcessor, dev);

    const size_t threads_per_block = 256;
    const size_t num_blocks        = static_cast<size_t>(sm_count)
                                     * (max_threads_per_sm / threads_per_block);  // 768 auf der 4090


//    cudaFuncAttributes attr;
//    cudaFuncGetAttributes(&attr, ci_kernel);
//    cudaFuncGetAttributes(&attr, mi_kernel);

    cudaFuncAttributes attr_ci, attr_mi;   // gegen Lazy Loading
    cudaFuncGetAttributes(&attr_ci, ci_kernel);
    cudaFuncGetAttributes(&attr_mi, mi_kernel);

    int occ_ci = 0, occ_mi = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_ci, ci_kernel, (int)threads_per_block, 0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_mi, mi_kernel, (int)threads_per_block, 0);

    std::printf("SMs %d, Threads/SM %d -> %zu Bloecke (%zu/SM)\n",
                sm_count, max_threads_per_sm, num_blocks, num_blocks / sm_count);
    std::printf("CI: %d Reg/Thread, %d Bloecke/SM resident\n", attr_ci.numRegs, occ_ci);
    std::printf("MI: %d Reg/Thread, %d Bloecke/SM resident\n", attr_mi.numRegs, occ_mi);

    state.exec([&](nvbench::launch& launch) {

        cudaStream_t current_stream = launch.get_stream();

        ci_kernel<<<num_blocks, threads_per_block, 0, current_stream>>>
                (d_ci_data, elements_ci, iterations_ci);

        mi_kernel<<<num_blocks, threads_per_block, 0, current_stream>>>
                (d_mi_in, d_mi_out, d_mi_indices, elements_mi, iterations_mi);


    });

    cudaFree(d_ci_data);
    cudaFree(d_mi_in);
    cudaFree(d_mi_out);
    cudaFree(d_mi_indices);

}


NVBENCH_BENCH(bench_baseline_0)
        .add_int64_axis("NumElems", {1024 * 1024 * 128})
        .add_int64_axis("Iterations_MI", {177})         // 100 auf RTX4090
        .add_int64_axis("Iterations_CI", {291'600});// 270'000 auf RTX4090
#include <nvbench/nvbench.cuh>
#include "kernels_strided.cuh"
#include <cuda/cmath>
#include <numeric>
#include <random>
#include <algorithm>

/*
 Ansatz 3: Warp-Level Task Assignment

 Die Aufgabe wird innerhalb des Blocks pro Warp zugewiesen:

   warp_id <  ci_warps  ->  CI
   warp_id >= ci_warps  ->  MI

 Die Verzweigung erfolgt entlang threadIdx.x / 32, ist also innerhalb eines
 Warps uniform ==> Keine Warp-Divergenz.

 Kein __syncthreads(): Weil CI- und MI-Warps haben stark unterschiedliche
 Laufzeiten
 */

__global__ void warp_split_kernel(float* ci_data, size_t n_ci, size_t it_ci,
                                  const float* mi_in, float* mi_out,
                                  const size_t* mi_indices, size_t n_mi, size_t it_mi,
                                  unsigned int ci_warps) {
    const unsigned int warps_per_block = blockDim.x / 32u;
    const unsigned int mi_warps        = warps_per_block - ci_warps;
    const unsigned int warp_id         = threadIdx.x / 32u;
    const unsigned int lane            = threadIdx.x % 32u;

    if (warp_id < ci_warps) {
        // Laufende Nummer dieses Warps unter allen CI-Warps des Grids
        const size_t task_warp  = static_cast<size_t>(blockIdx.x) * ci_warps + warp_id;
        const size_t task_warps = static_cast<size_t>(gridDim.x) * ci_warps;
        compute_ci_fma(ci_data, n_ci, it_ci,
                       task_warp * 32u + lane,
                       task_warps * 32u);
    } else {
        const size_t task_warp  = static_cast<size_t>(blockIdx.x) * mi_warps
                                  + (warp_id - ci_warps);
        const size_t task_warps = static_cast<size_t>(gridDim.x) * mi_warps;
        compute_mi_indirect(mi_in, mi_out, mi_indices, n_mi, it_mi,
                            task_warp * 32u + lane,
                            task_warps * 32u);
    }
}

void bench_approach_3(nvbench::state& state) {
    const size_t num_elems     = state.get_int64("NumElems");
    const size_t iterations_mi = state.get_int64("Iterations_MI");
    const size_t iterations_ci = state.get_int64("Iterations_CI");
    const auto   ci_warps      = static_cast<unsigned int>(state.get_int64("CI_Warps_Per_Block"));

    // Grid zur Laufzeit ableiten statt fester 4090-Konstanten
    int dev = 0;
    cudaGetDevice(&dev);
    int sm_count = 0, max_threads_per_sm = 0;
    cudaDeviceGetAttribute(&sm_count,           cudaDevAttrMultiProcessorCount,         dev);
    cudaDeviceGetAttribute(&max_threads_per_sm, cudaDevAttrMaxThreadsPerMultiProcessor, dev);

    const size_t threads_per_block = 256;
    const size_t num_blocks        = static_cast<size_t>(sm_count)
                                     * (max_threads_per_sm / threads_per_block);  // 768 auf der 4090
    const unsigned int warps_per_block = threads_per_block / 32;                // 8

    if (ci_warps == 0 || ci_warps >= warps_per_block) {
        state.skip("CI_Warps_Per_Block muss zwischen 1 und warps_per_block-1 liegen");
        return;
    }

    // 1. Host-Array erstellen und mit 0, 1, 2, ..., N-1 füllen
    std::vector<size_t> h_mi_indices(num_elems);
    std::iota(h_mi_indices.begin(), h_mi_indices.end(), 0);

    // 2. Array zufällig mischen (Permutation)
    std::mt19937 g(42); // Fester Seed für Reproduzierbarkeit
    std::shuffle(h_mi_indices.begin(), h_mi_indices.end(), g);

    const size_t elements_ci = num_elems;
    const size_t elements_mi = num_elems;

    float*  d_ci_data    = nullptr;
    float*  d_mi_in      = nullptr;
    float*  d_mi_out     = nullptr;
    size_t* d_mi_indices = nullptr;

    cudaMalloc(&d_ci_data,    elements_ci * sizeof(float));
    cudaMalloc(&d_mi_in,      elements_mi * sizeof(float));
    cudaMalloc(&d_mi_out,     elements_mi * sizeof(float));
    cudaMalloc(&d_mi_indices, elements_mi * sizeof(size_t));

    cudaMemset(d_ci_data, 0, elements_ci * sizeof(float));
    cudaMemset(d_mi_in,   0, elements_mi * sizeof(float));
    cudaMemset(d_mi_out,  0, elements_mi * sizeof(float));
    cudaMemcpy(d_mi_indices, h_mi_indices.data(),
               elements_mi * sizeof(size_t), cudaMemcpyHostToDevice);

    cudaFuncAttributes attr;  // gegen LazyLoads
    cudaFuncGetAttributes(&attr, warp_split_kernel);
    const unsigned int blocks_per_sm = max_threads_per_sm / threads_per_block;
    int occ = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, warp_split_kernel, (int)threads_per_block, 0);
    std::printf("CI-Warps/Block %u von %u -> pro SM %u CI-Warps / %u MI-Warps, %d Register/Thread, %d Bloecke/SM resident\n",
                ci_warps, warps_per_block,
                ci_warps * blocks_per_sm, (warps_per_block - ci_warps) * blocks_per_sm,
                attr.numRegs, occ);


    state.exec([&](nvbench::launch& launch) {
        warp_split_kernel<<<num_blocks, threads_per_block, 0, launch.get_stream()>>>(
                d_ci_data, elements_ci, iterations_ci,
                d_mi_in, d_mi_out, d_mi_indices, elements_mi, iterations_mi,
                ci_warps);
    });

    cudaFree(d_ci_data);
    cudaFree(d_mi_in);
    cudaFree(d_mi_out);
    cudaFree(d_mi_indices);
}

/* ---RTX 4090
 * NVBENCH_BENCH(bench_approach_3)
        .add_int64_axis("NumElems", {1024 * 1024 * 128})
        .add_int64_axis("Iterations_MI", {100})
        .add_int64_axis("Iterations_CI", {270'000})
        .add_int64_axis("CI_Warps_Per_Block", {1, 2, 3, 4, 5, 6, 7});
*/

NVBENCH_BENCH(bench_approach_3)
        .add_int64_axis("NumElems", {1024 * 1024 * 128})
        .add_int64_axis("Iterations_MI", {177})
        .add_int64_axis("Iterations_CI", {291'600})
        .add_int64_axis("CI_Warps_Per_Block", {1, 2, 3, 4, 5, 6, 7});
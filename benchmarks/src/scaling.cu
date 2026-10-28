#include <nvbench/nvbench.cuh>
#include "kernels_strided.cuh"
#include <cuda/cmath>
#include <numeric>
#include <random>
#include <algorithm>

/*
 Skalierungsmessung: Laufzeit jedes Kernels allein über der Blockzahl.
*/


// Die Indexarrays sind bei 134 M Elementen teuer und hängen nur von
// (num_elems, permuted) ab. --> einmal erzeugen und wiederverwenden
static const std::vector<size_t>& get_indices(size_t num_elems, bool permuted) {
    static std::vector<size_t> cache;
    static size_t cached_n = 0;
    static bool   cached_p = false;
    static bool   valid    = false;

    if (!valid || cached_n != num_elems || cached_p != permuted) {
        cache.resize(num_elems);
        std::iota(cache.begin(), cache.end(), 0);
        if (permuted) {
            std::mt19937 g(42); // Fester Seed für Reproduzierbarkeit
            std::shuffle(cache.begin(), cache.end(), g);
        }
        cached_n = num_elems;
        cached_p = permuted;
        valid    = true;
    }
    return cache;
}

void bench_scaling_ci(nvbench::state& state) {
    const size_t num_elems     = state.get_int64("NumElems");
    const size_t iterations_ci = state.get_int64("Iterations_CI");
    const size_t num_blocks    = state.get_int64("NumBlocks");

    const size_t threads_per_block = 256;

    float* d_ci_data = nullptr;
    cudaMalloc(&d_ci_data, num_elems * sizeof(float));
    cudaMemset(d_ci_data, 0, num_elems * sizeof(float));

    cudaFuncAttributes attr;  // gegen LazyLoads
    cudaFuncGetAttributes(&attr, ci_kernel);
    std::printf("CI: %4zu Blöcke (%zu Threads), %zu Elemente, %d Register/Thread\n",
                num_blocks, num_blocks * threads_per_block, num_elems, attr.numRegs);

    state.exec([&](nvbench::launch& launch) {
        ci_kernel<<<num_blocks, threads_per_block, 0, launch.get_stream()>>>(
                d_ci_data, num_elems, iterations_ci);
    });

    cudaFree(d_ci_data);
}

void bench_scaling_mi(nvbench::state& state) {
    const size_t num_elems     = state.get_int64("NumElems");
    const size_t iterations_mi = state.get_int64("Iterations_MI");
    const size_t num_blocks    = state.get_int64("NumBlocks");
    const bool   permuted      = state.get_int64("Permuted") != 0;

    const size_t threads_per_block = 256;

    const std::vector<size_t>& h_mi_indices = get_indices(num_elems, permuted);

    float*  d_mi_in      = nullptr;
    float*  d_mi_out     = nullptr;
    size_t* d_mi_indices = nullptr;

    cudaMalloc(&d_mi_in,      num_elems * sizeof(float));
    cudaMalloc(&d_mi_out,     num_elems * sizeof(float));
    cudaMalloc(&d_mi_indices, num_elems * sizeof(size_t));

    cudaMemset(d_mi_in,  0, num_elems * sizeof(float));
    cudaMemset(d_mi_out, 0, num_elems * sizeof(float));
    cudaMemcpy(d_mi_indices, h_mi_indices.data(),
               num_elems * sizeof(size_t), cudaMemcpyHostToDevice);

    cudaFuncAttributes attr;  // gegen LazyLoads
    cudaFuncGetAttributes(&attr, mi_kernel);
    std::printf("MI: %4zu Blöcke (%zu Threads), %zu Elemente, %s, %d Register/Thread\n",
                num_blocks, num_blocks * threads_per_block, num_elems,
                permuted ? "permutiert" : "sequenziell", attr.numRegs);

    state.exec([&](nvbench::launch& launch) {
        mi_kernel<<<num_blocks, threads_per_block, 0, launch.get_stream()>>>(
                d_mi_in, d_mi_out, d_mi_indices, num_elems, iterations_mi);
    });

    cudaFree(d_mi_in);
    cudaFree(d_mi_out);
    cudaFree(d_mi_indices);
}


/* --- RTX4090 Parameter
NVBENCH_BENCH(bench_scaling_ci)
        .add_int64_axis("NumElems", {1024 * 1024 * 128})
        .add_int64_axis("Iterations_CI", {270'000})
        .add_int64_axis("NumBlocks", {16, 32, 64, 128, 256, 384, 512, 640, 768});

NVBENCH_BENCH(bench_scaling_mi)
        .add_int64_axis("NumElems", {1024 * 1024 * 128})
        .add_int64_axis("Iterations_MI", {100})
        .add_int64_axis("NumBlocks", {16, 32, 64, 128, 256, 384, 512, 640, 768})
        .add_int64_axis("Permuted", {0, 1});
*/

// RTX PRO 6000: 188 SMs, max. 6 Bloecke/SM (1536 Threads)
// Vielfache von 188 = 1..6 Bloecke/SM, darunter weniger SMs als Bloecke
NVBENCH_BENCH(bench_scaling_ci)
        .add_int64_axis("NumElems", {1024 * 1024 * 128})
        .add_int64_axis("Iterations_CI", {291'600})
        .add_int64_axis("NumBlocks", {12, 24, 48, 94, 188, 376, 564, 752, 940, 1128});

NVBENCH_BENCH(bench_scaling_mi)
        .add_int64_axis("NumElems", {1024 * 1024 * 128})
        .add_int64_axis("Iterations_MI", {177})
        .add_int64_axis("NumBlocks", {12, 24, 48, 94, 188, 376, 564, 752, 940, 1128})
        .add_int64_axis("Permuted", {0, 1});
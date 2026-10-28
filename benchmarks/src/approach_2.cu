#include <nvbench/nvbench.cuh>
#include "kernels_strided.cuh"
#include <cuda/cmath>
#include <numeric>
#include <random>
#include <algorithm>

/* Ansatz 2: Hybrides SM-Throttling (SM-zentrisch, nach Paper von Wu et al.)

 */


__device__ __forceinline__ unsigned int get_smid() {
    unsigned int smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    return smid;
}

__global__ void sm_throttle_kernel(float* ci_data, size_t n_ci, size_t it_ci,
                                   const float* mi_in, float* mi_out,
                                   const size_t* mi_indices, size_t n_mi, size_t it_mi,
                                   unsigned int* sm_slot_counter,
                                   unsigned int* ci_task_counter,
                                   unsigned int* mi_task_counter,
                                   unsigned int  ci_per_sm,
                                   size_t blocks_ci, size_t blocks_mi) {
    __shared__ unsigned int s_is_ci;
    __shared__ unsigned int s_task_block;

    if (threadIdx.x == 0) {
        const unsigned int smid = get_smid();
        const unsigned int slot = atomicAdd(&sm_slot_counter[smid], 1u);

        if (slot < ci_per_sm) {
            s_is_ci      = 1u;
            s_task_block = atomicAdd(ci_task_counter, 1u);
        } else {
            s_is_ci      = 0u;
            s_task_block = atomicAdd(mi_task_counter, 1u);
        }
    }
    __syncthreads();

    const size_t rank = static_cast<size_t>(s_task_block) * blockDim.x + threadIdx.x;

    if (s_is_ci) {
        compute_ci_fma(ci_data, n_ci, it_ci, rank, blocks_ci * blockDim.x);
    } else {
        compute_mi_indirect(mi_in, mi_out, mi_indices, n_mi, it_mi,
                            rank, blocks_mi * blockDim.x);
    }
}

// Die Permutation ist bei 134 M Elementen teuer und hängt nur von num_elems ab.
// Einmal erzeugen und über alle Konfigurationen wiederverwenden spart bei
// 25 Konfigurationen mehrere Minuten reine Hostzeit.
static const std::vector<size_t>& get_permutation(size_t num_elems) {
    static std::vector<size_t> cache;
    static size_t cached_n = 0;
    if (cached_n != num_elems) {
        cache.resize(num_elems);
        std::iota(cache.begin(), cache.end(), 0);
        std::mt19937 g(42); // Fester Seed für Reproduzierbarkeit
        std::shuffle(cache.begin(), cache.end(), g);
        cached_n = num_elems;
    }
    return cache;
}

void bench_approach_2(nvbench::state& state) {
    const size_t num_elems     = state.get_int64("NumElems");
    const size_t iterations_mi = state.get_int64("Iterations_MI");
    const size_t iterations_ci = state.get_int64("Iterations_CI");
    const auto   mi_per_sm     = static_cast<unsigned int>(state.get_int64("MI_Blocks_Per_SM"));

    // Grid zur Laufzeit ableiten statt fester 4090-Konstanten
    int dev = 0;
    cudaGetDevice(&dev);
    int sm_count = 0, max_threads_per_sm = 0;
    cudaDeviceGetAttribute(&sm_count,           cudaDevAttrMultiProcessorCount,         dev);
    cudaDeviceGetAttribute(&max_threads_per_sm, cudaDevAttrMaxThreadsPerMultiProcessor, dev);

    const size_t threads_per_block = 256;
    const auto   blocks_per_sm     = static_cast<unsigned int>(max_threads_per_sm / threads_per_block);
    const size_t num_blocks        = static_cast<size_t>(sm_count) * blocks_per_sm;

    if (mi_per_sm >= blocks_per_sm) {
        state.skip("MI_Blocks_Per_SM muss kleiner als Blöcke/SM sein");
        return;
    }
    const unsigned int ci_per_sm = blocks_per_sm - mi_per_sm;
    const size_t blocks_ci = static_cast<size_t>(sm_count) * ci_per_sm;
    const size_t blocks_mi = static_cast<size_t>(sm_count) * mi_per_sm;

    const std::vector<size_t>& h_mi_indices = get_permutation(num_elems);

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

    // Zähler: pro SM einer, plus je einer für die Task-Vergabe. Großzügig
    // dimensioniert, falls %smid größer als sm_count wird.
    const size_t counter_slots = 1024;
    unsigned int* d_counters = nullptr;
    cudaMalloc(&d_counters, (counter_slots + 2) * sizeof(unsigned int));
    unsigned int* d_sm_slot = d_counters;
    unsigned int* d_ci_task = d_counters + counter_slots;
    unsigned int* d_mi_task = d_counters + counter_slots + 1;

    cudaFuncAttributes attr;  // gegen LazyLoads
    cudaFuncGetAttributes(&attr, sm_throttle_kernel);

    // Die Invariante "jeder SM bekommt genau blocks_per_sm Blöcke" gilt nur,
    // wenn wirklich alle Blöcke gleichzeitig resident sind.
    int achievable = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &achievable, sm_throttle_kernel, static_cast<int>(threads_per_block), 0);
    if (achievable < static_cast<int>(blocks_per_sm)) {
        std::printf("WARNUNG: nur %d von %u Blöcken/SM resident (%d Register/Thread). "
                    "Slot-Vergabe ist nicht mehr garantiert.\n",
                    achievable, blocks_per_sm, attr.numRegs);
        state.skip("Occupancy zu niedrig für garantierte Slot-Vergabe");
        cudaFree(d_counters);
        return;
    }

/*
 * RTX 4090:
 *  // Referenz aus der Kalibrierungsmessung: 270.000 Iterationen ergeben
    // 3,585 s CI solo, Iterations_MI = 100 ergibt 3,354 s MI solo.
    const double ratio = (static_cast<double>(iterations_ci) / 270000.0 * 3.585) / 3.354;
    std::printf("Iter_CI %7zu (CI:MI etwa %.2f), MI/SM %u -> %4zu CI-Blöcke / %4zu MI-Blöcke, "
                "%d Register/Thread\n",
                iterations_ci, ratio, mi_per_sm, blocks_ci, blocks_mi, attr.numRegs);
*/

    // Referenz PRO 6000: 291.600 Iterationen ergeben 2,994132 s CI solo,
    // Iterations_MI = 177 ergibt 2,994080 s MI solo.
    const double ratio = (static_cast<double>(iterations_ci) / 291600.0 * 2.994132) / 2.994080;
    std::printf("Iter_CI %7zu (CI:MI etwa %.2f), MI/SM %u -> %4zu CI-Blöcke / %4zu MI-Blöcke, "
                "%d Register/Thread\n",
                iterations_ci, ratio, mi_per_sm, blocks_ci, blocks_mi, attr.numRegs);

    state.exec([&](nvbench::launch& launch) {
        // Zähler vor jedem Launch zurücksetzen. 4 KB Memset gegenüber
        // mehreren Sekunden Kernellaufzeit ist vernachlässigbar.
        cudaMemsetAsync(d_counters, 0, (counter_slots + 2) * sizeof(unsigned int),
                        launch.get_stream());
        sm_throttle_kernel<<<num_blocks, threads_per_block, 0, launch.get_stream()>>>(
                d_ci_data, elements_ci, iterations_ci,
                d_mi_in, d_mi_out, d_mi_indices, elements_mi, iterations_mi,
                d_sm_slot, d_ci_task, d_mi_task,
                ci_per_sm, blocks_ci, blocks_mi);
    });

    cudaFree(d_counters);
    cudaFree(d_ci_data);
    cudaFree(d_mi_in);
    cudaFree(d_mi_out);
    cudaFree(d_mi_indices);
}
// ---RTX 4090
// Iterations_CI ändert das Arbeitsverhältnis CI:MI !
//  vermutlich (aus Erfahrung) 67.500 -> 0,27, 135.000 -> 0,53, 270.000 -> 1,07, 405.000 -> 1,60, 540.000 -> 2,14

// ---RTX 6000
// Verhältnis CI:MI auf der PRO 6000 (291.600 -> 1,00):
//   72.900 -> 0,25   145.800 -> 0,50   291.600 -> 1,00   437.400 -> 1,50   583.200 -> 2,00
NVBENCH_BENCH(bench_approach_2)
        .add_int64_axis("NumElems", {1024 * 1024 * 128})
        .add_int64_axis("Iterations_MI", {177})
        .add_int64_axis("Iterations_CI", {72'900, 145'800, 291'600, 437'400, 583'200})
        .add_int64_axis("MI_Blocks_Per_SM", {1, 2, 3, 4, 5});
#include <nvbench/nvbench.cuh>
#include "../include/kernels_strided.cuh"
#include <cuda/cmath>
#include <numeric>
#include <random>
#include <algorithm>
#include <map>


/*
 Ansatz 1: Naive Kernel Fusion (Uberkernel)

 Beide Aufgaben laufen in einem (!) Grid. Gerade Blöcke rechnen CI, ungerade MI.
 Ob CI- und MI-Blöcke dadurch tatsaechlich auf denselben SMs landen,
 entscheidet der Hardware-Scheduler.
*/



__device__ __forceinline__ unsigned int get_smid() {
    unsigned int smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    return smid;
}

__global__ void fused_kernel(float* ci_data, size_t n_ci, size_t it_ci,
                             const float* mi_in, float* mi_out, const size_t* mi_indices,
                             size_t n_mi, size_t it_mi,
                             unsigned int* smid_out) {
    const size_t task_blocks = gridDim.x / 2;      // Blöcke je Aufgabe
    const size_t task_block  = blockIdx.x / 2;     // Index innerhalb der Aufgabe

    const size_t rank  = task_block * blockDim.x + threadIdx.x;
    const size_t count = task_blocks * blockDim.x;

    if (threadIdx.x == 0 && smid_out != nullptr) {
        smid_out[blockIdx.x] = get_smid();
    }

    // Verzweigung auf Blockebene, innerhalb eines Warps also keine Divergenz.
    if (blockIdx.x % 2 == 0) {
        compute_ci_fma(ci_data, n_ci, it_ci, rank, count);
    } else {
        compute_mi_indirect(mi_in, mi_out, mi_indices, n_mi, it_mi, rank, count);
    }
}

// Host-seitige Auswertung: Wie viele CI- und MI-Blöcke hat jeder SM bekommen?
static void report_sm_distribution(const std::vector<unsigned int>& smid) {
    struct SmStats { unsigned int ci = 0, mi = 0; };
    std::map<unsigned int, SmStats> per_sm;

    unsigned int ungeschrieben = 0;
    for (std::size_t b = 0; b < smid.size(); ++b) {
        if (smid[b] == 0xFFFFFFFFu) { ++ungeschrieben; continue; }
        if (b % 2 == 0) per_sm[smid[b]].ci++;
        else            per_sm[smid[b]].mi++;
    }

    unsigned int gemischt = 0, nur_ci = 0, nur_mi = 0;
    std::map<unsigned int, unsigned int> hist_ci;   // CI-Blöcke pro SM -> Anzahl SMs
    for (const auto& kv : per_sm) {
        const auto& s = kv.second;
        if (s.ci > 0 && s.mi > 0) ++gemischt;
        else if (s.ci > 0)        ++nur_ci;
        else                      ++nur_mi;
        hist_ci[s.ci]++;
    }

    std::printf("\n--- SM-Verteilung (blockIdx.x %% 2) ---\n");
    std::printf("SMs belegt      : %zu\n", per_sm.size());
    std::printf("SMs mit BEIDEN  : %u\n", gemischt);
    std::printf("SMs nur CI      : %u\n", nur_ci);
    std::printf("SMs nur MI      : %u\n", nur_mi);
    if (ungeschrieben > 0)
        std::printf("WARNUNG: %u Blöcke ohne SM-ID\n", ungeschrieben);

    std::printf("\nHistogramm CI-Blöcke pro SM:\n");
    for (const auto& kv : hist_ci)
        std::printf("  %u CI-Blöcke -> %u SMs\n", kv.first, kv.second);

    std::printf("\nAlle SMs [smid: CI/MI]:\n");
    int n = 0;
    for (const auto& kv : per_sm) {
        std::printf("[%3u: %u/%u] ", kv.first, kv.second.ci, kv.second.mi);
        if (++n % 8 == 0) std::printf("\n");
    }
    if (n % 8 != 0) std::printf("\n");
    std::printf("\n");
}

void bench_approach_1(nvbench::state& state) {
    const size_t num_elems     = state.get_int64("NumElems");
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

    // Grid zur Laufzeit ableiten statt fester 4090-Konstanten
    int dev = 0;
    cudaGetDevice(&dev);
    int sm_count = 0, max_threads_per_sm = 0;
    cudaDeviceGetAttribute(&sm_count,           cudaDevAttrMultiProcessorCount,         dev);
    cudaDeviceGetAttribute(&max_threads_per_sm, cudaDevAttrMaxThreadsPerMultiProcessor, dev);

    const size_t threads_per_block = 256;
    const size_t num_blocks        = static_cast<size_t>(sm_count)
                                     * (max_threads_per_sm / threads_per_block);  // 768 auf der 4090

    unsigned int* d_smid = nullptr;
    cudaMalloc(&d_smid, num_blocks * sizeof(unsigned int));
    cudaMemset(d_smid, 0xFF, num_blocks * sizeof(unsigned int));

    cudaFuncAttributes attr;  // gegen LazyLoads
    cudaFuncGetAttributes(&attr, fused_kernel);
    int occ = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, fused_kernel, (int)threads_per_block, 0);
    std::printf("fused_kernel: %d Register/Thread, %d Bloecke/SM resident, Grid = %zu Bloecke auf %d SMs\n",
                attr.numRegs, occ, num_blocks, sm_count);

    state.exec([&](nvbench::launch& launch) {
        fused_kernel<<<num_blocks, threads_per_block, 0, launch.get_stream()>>>(
                d_ci_data, elements_ci, iterations_ci,
                d_mi_in, d_mi_out, d_mi_indices, elements_mi, iterations_mi,
                d_smid);
    });

    // Auswertung nach dem Benchmark, zeigt den letzten Durchlauf.
    std::vector<unsigned int> h_smid(num_blocks);
    cudaMemcpy(h_smid.data(), d_smid, num_blocks * sizeof(unsigned int),
               cudaMemcpyDeviceToHost);
    report_sm_distribution(h_smid);

    // Rohdaten für eigene Auswertung
    std::FILE* f = std::fopen("approach_1_smid.csv", "w");
    if (f) {
        std::fprintf(f, "block_id,smid,is_ci\n");
        for (size_t b = 0; b < num_blocks; ++b)
            std::fprintf(f, "%zu,%u,%d\n", b, h_smid[b], (b % 2 == 0) ? 1 : 0);
        std::fclose(f);
        std::printf("Rohdaten: approach_1_smid.csv\n");
    }

    cudaFree(d_smid);
    cudaFree(d_ci_data);
    cudaFree(d_mi_in);
    cudaFree(d_mi_out);
    cudaFree(d_mi_indices);
}

/* --- RTX 4090 Parameter
NVBENCH_BENCH(bench_approach_1)
        .add_int64_axis("NumElems", {1024 * 1024 * 128})
        .add_int64_axis("Iterations_MI", {100})
        .add_int64_axis("Iterations_CI", {270'000});
*/

NVBENCH_BENCH(bench_approach_1)
        .add_int64_axis("NumElems", {1024 * 1024 * 128})
        .add_int64_axis("Iterations_MI", {177})
        .add_int64_axis("Iterations_CI", {291'600});

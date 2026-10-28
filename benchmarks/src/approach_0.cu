#include <nvbench/nvbench.cuh>
#include "kernels_strided.cuh"
#include <cuda/cmath>
#include <numeric>
#include <random>
#include <algorithm>

/*
 Ansatz 0: Strikte SM-Isolation über Green Contexts

 Neu gegenüber der ersten Fassung:
   - CI_SM_Percent ist eine Achse (vorher fest 96 %)
   - Die tatsächlich zugeteilten SM-Zahlen werden aus split_result gelesen
   - max_blocks_per_sm wird zur Laufzeit abgeleitet statt fest 6
*/

void bench_approach_0(nvbench::state& state) {

    const size_t num_elems = state.get_int64("NumElems");
    const size_t iterations_mi = state.get_int64("Iterations_MI");
    const size_t iterations_ci = state.get_int64("Iterations_CI");
    const auto   ci_percent = static_cast<unsigned int>(state.get_int64("CI_SM_Percent"));


    // 1. Host-Array erstellen und mit 0, 1, 2, ..., N-1 fuellen
    std::vector<size_t> h_mi_indices(num_elems);
    std::iota(h_mi_indices.begin(), h_mi_indices.end(), 0);

    // 2. Array zufaellig mischen (Permutation)
    std::mt19937 g(42); // Fester Seed fuer Reproduzierbarkeit
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

    // -------------------------------------------------------------------------
    // --------------------- Code zu Green Contexts Anfang ---------------------
    // https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/green-contexts.html#green-contexts-ease-of-use

    int gpu_device_index = 0;
    cudaSetDevice(gpu_device_index);

    // Get all available GPU SM resources
    cudaDevResource initial_SM_resources {};
    cudaDeviceGetDevResource(gpu_device_index, &initial_SM_resources, cudaDevResourceTypeSm);

    // Split SM resources. Should be a multiple of 2
    unsigned int sm_total = initial_SM_resources.sm.smCount;
    unsigned int sm_ci = (sm_total * ci_percent / 100u) & ~1u;
    if (sm_ci < 2u)            sm_ci = 2u;
    if (sm_ci > sm_total - 2u) sm_ci = sm_total - 2u;
    unsigned int sm_mi = sm_total - sm_ci;

    cudaDevResource split_result[2] {{}, {}}; // Warum nicht cudaDevSmResource? --> Dokumentation ist falsch!
    cudaDevSmResourceGroupParams group_params[2] =  {
            {.smCount=sm_ci, .coscheduledSmCount=0, .preferredCoscheduledSmCount=0, .flags=0},
            {.smCount=sm_mi, .coscheduledSmCount=0, .preferredCoscheduledSmCount=0, .flags=0}};
    cudaDevSmResourceSplit(&split_result[0],
                           2,
                           &initial_SM_resources,
                           nullptr,
                           0,
                           &group_params[0]);

    // Was der Treiber tatsächlich zugeteilt hat.
    const unsigned int actual_sm_ci = split_result[0].sm.smCount;
    const unsigned int actual_sm_mi = split_result[1].sm.smCount;

    // Work Queue Konfiguration definieren
    cudaDevResource wq_config {};
    wq_config.type = cudaDevResourceTypeWorkqueueConfig;
    wq_config.wqConfig.device = gpu_device_index;
    wq_config.wqConfig.sharingScope = cudaDevWorkqueueConfigScopeGreenCtxBalanced;
    // Signalisiert dem Treiber, dass wir 2 parallele Streams erwarten
    wq_config.wqConfig.wqConcurrencyLimit = 2;

    // SM-Ressourcen und WQ-Konfiguration für die Deskriptoren bündeln
    cudaDevResource resources_ci[2] = {split_result[0], wq_config};
    cudaDevResource resources_mi[2] = {split_result[1], wq_config};

    // Generate resource descriptors for each resource (Jetzt mit 2 Ressourcen pro Descriptor)
    cudaDevResourceDesc_t descr_ci {};
    cudaDevResourceDesc_t descr_mi {};
    cudaDevResourceGenerateDesc(&descr_ci, resources_ci, 2);
    cudaDevResourceGenerateDesc(&descr_mi, resources_mi, 2);


    // Create green contexts
    cudaExecutionContext_t green_ctx_ci {};
    cudaExecutionContext_t green_ctx_mi {};
    cudaGreenCtxCreate(&green_ctx_ci, descr_ci, gpu_device_index, 0);
    cudaGreenCtxCreate(&green_ctx_mi, descr_mi, gpu_device_index, 0);

    // --------------------- Code zu Green Contexts Ende -----------------------
    // -------------------------------------------------------------------------


    cudaStream_t stream_ci;
    cudaStream_t stream_mi;

    cudaExecutionCtxStreamCreate(&stream_ci, green_ctx_ci, cudaStreamDefault, 0);
    cudaExecutionCtxStreamCreate(&stream_mi, green_ctx_mi, cudaStreamDefault, 0);


    cudaEvent_t start_event;
    cudaEvent_t end_ci_event;
    cudaEvent_t end_mi_event;

    cudaEventCreate(&start_event);
    cudaEventCreate(&end_ci_event);
    cudaEventCreate(&end_mi_event);

    // Blockzahl aus den tatsächlich zugeteilten SMs ableiten
    int max_threads_per_sm = 0;
    cudaDeviceGetAttribute(&max_threads_per_sm, cudaDevAttrMaxThreadsPerMultiProcessor,
                           gpu_device_index);

    const size_t threads_per_block = 256;
    const size_t max_blocks_per_sm = max_threads_per_sm / threads_per_block;  // 6 auf der 4090
    const size_t num_blocks_ci = actual_sm_ci * max_blocks_per_sm;
    const size_t num_blocks_mi = actual_sm_mi * max_blocks_per_sm;

    std::printf("Split %2u%%: angefordert %u/%u SMs -> zugeteilt %u/%u SMs "
                "(%zu CI-Bloecke / %zu MI-Bloecke)\n",
                ci_percent, sm_ci, sm_mi, actual_sm_ci, actual_sm_mi,
                num_blocks_ci, num_blocks_mi);

    cudaFuncAttributes attr;  // gegen LazyLoads
    cudaFuncGetAttributes(&attr, ci_kernel);
    cudaFuncGetAttributes(&attr, mi_kernel);


    state.exec([&](nvbench::launch& launch) {
        cudaStream_t bench_stream = launch.get_stream();

        cudaEventRecord(start_event, bench_stream);

        cudaStreamWaitEvent(stream_ci, start_event, 0);
        cudaStreamWaitEvent(stream_mi, start_event, 0);

        ci_kernel<<<num_blocks_ci, threads_per_block, 0, stream_ci>>>
                (d_ci_data, elements_ci, iterations_ci);

        mi_kernel<<<num_blocks_mi, threads_per_block, 0, stream_mi>>>
                (d_mi_in, d_mi_out, d_mi_indices, elements_mi, iterations_mi);

        cudaEventRecord(end_ci_event, stream_ci);
        cudaEventRecord(end_mi_event, stream_mi);

        cudaStreamWaitEvent(bench_stream, end_ci_event, 0);
        cudaStreamWaitEvent(bench_stream, end_mi_event, 0);

    });

    cudaStreamDestroy(stream_ci);
    cudaStreamDestroy(stream_mi);
    cudaExecutionCtxDestroy(green_ctx_ci);
    cudaExecutionCtxDestroy(green_ctx_mi);
    cudaEventDestroy(start_event);
    cudaEventDestroy(end_ci_event);
    cudaEventDestroy(end_mi_event);

    cudaFree(d_ci_data);
    cudaFree(d_mi_in);
    cudaFree(d_mi_out);
    cudaFree(d_mi_indices);

}

// RTX PRO 6000 (188 SMs), angefordert CI/MI:
//   50% -> 94/94    70% -> 130/58   80% -> 150/38   90% -> 168/20
//   94% -> 176/12   96% -> 180/8    98% -> 184/4    99% -> 186/2
NVBENCH_BENCH(bench_approach_0)
        .add_int64_axis("NumElems", {1024 * 1024 * 128})
        .add_int64_axis("Iterations_MI", {177})
        .add_int64_axis("Iterations_CI", {291'600})
//        .add_int64_axis("CI_SM_Percent", {50, 70, 80, 90, 94, 96, 98, 99});
        .add_int64_axis("CI_SM_Percent", {78, 79, 81, 82, 83, 85, 86, 87});
    // 81 -> 152/36  82 -> 154/34  83 -> 156/32  85 -> 158/30
    // 86 -> 160/28  87 -> 162/26  88 -> 164/24  89 -> 166/22
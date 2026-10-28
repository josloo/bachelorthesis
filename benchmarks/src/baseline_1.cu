#include <nvbench/nvbench.cuh>
#include "kernels_strided.cuh"
#include <cuda/cmath>
#include <numeric>
#include <random>
#include <algorithm>

void bench_baseline_1(nvbench::state& state) {

    const size_t num_elems = state.get_int64("NumElems");
    const size_t iterations_mi = state.get_int64("Iterations_MI");
    const size_t iterations_ci = state.get_int64("Iterations_CI");
//    const bool mi_first = state.get_int64("MI_First") == 1;


    // 1. Host-Array erstellen und mit 0, 1, 2, ..., N-1 füllen
    std::vector<size_t> h_mi_indices(num_elems);
    std::iota(h_mi_indices.begin(), h_mi_indices.end(), 0);

    // 2. Array zufällig mischen (Permutation)
    std::mt19937 g(42); // Fester Seed für Reproduzierbarkeit
    std::shuffle(h_mi_indices.begin(), h_mi_indices.end(), g);


    const size_t elements_ci = num_elems;
    const size_t elements_mi = num_elems; //* iterations_mi;

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

    cudaStream_t stream_ci;
    cudaStream_t stream_mi;

    cudaStreamCreateWithFlags(&stream_ci, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&stream_mi, cudaStreamNonBlocking);

    cudaEvent_t start_event;
    cudaEvent_t end_ci_event;
    cudaEvent_t end_mi_event;

    cudaEventCreate(&start_event);
    cudaEventCreate(&end_ci_event);
    cudaEventCreate(&end_mi_event);

    const size_t threads_per_block = 256;

    int dev = 0;
    cudaGetDevice(&dev);
    int sm_count = 0, max_threads_per_sm = 0;
    cudaDeviceGetAttribute(&sm_count,           cudaDevAttrMultiProcessorCount,         dev);
    cudaDeviceGetAttribute(&max_threads_per_sm, cudaDevAttrMaxThreadsPerMultiProcessor, dev);
    const size_t num_blocks = static_cast<size_t>(sm_count) * (max_threads_per_sm / threads_per_block);


    cudaFuncAttributes attr_ci, attr_mi;   // gegen Lazy Loading
    cudaFuncGetAttributes(&attr_ci, ci_kernel);
    cudaFuncGetAttributes(&attr_mi, mi_kernel);
    std::printf("SMs %d -> %zu Bloecke, CI %d Reg, MI %d Reg\n",
                sm_count, num_blocks, attr_ci.numRegs, attr_mi.numRegs);

    state.exec([&](nvbench::launch& launch) {
        cudaStream_t bench_stream = launch.get_stream();

        cudaEventRecord(start_event, bench_stream);

        cudaStreamWaitEvent(stream_ci, start_event, 0);
        cudaStreamWaitEvent(stream_mi, start_event, 0);


        ci_kernel<<<num_blocks, threads_per_block, 0, stream_ci>>>
                (d_ci_data, elements_ci, iterations_ci);
        mi_kernel<<<num_blocks, threads_per_block, 0, stream_mi>>>
                (d_mi_in, d_mi_out, d_mi_indices, elements_mi, iterations_mi);


        cudaEventRecord(end_ci_event, stream_ci);
        cudaEventRecord(end_mi_event, stream_mi);

        cudaStreamWaitEvent(bench_stream, end_ci_event, 0);
        cudaStreamWaitEvent(bench_stream, end_mi_event, 0);

    });

    cudaStreamDestroy(stream_ci);
    cudaStreamDestroy(stream_mi);
    cudaEventDestroy(start_event);
    cudaEventDestroy(end_ci_event);
    cudaEventDestroy(end_mi_event);

    cudaFree(d_ci_data);
    cudaFree(d_mi_in);
    cudaFree(d_mi_out);
    cudaFree(d_mi_indices);

}

NVBENCH_BENCH(bench_baseline_1)
        .add_int64_axis("NumElems", {1024 * 1024 * 128})
        .add_int64_axis("Iterations_MI", {177})
        .add_int64_axis("Iterations_CI", {291'600});
//        .add_int64_axis("MI_First", {0,1});
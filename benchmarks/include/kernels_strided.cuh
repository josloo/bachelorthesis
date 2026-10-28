#pragma once
#include <cstddef>
#include <cuda_runtime.h>
//#include <math_functions.h>

/*

   rank  = laufende Nummer dieses Threads innerhalb der Threads, die diese
           Aufgabe bearbeiten
   count = Gesamtzahl der Threads, die diese Aufgabe bearbeiten

 Bei den Baselines und Ansatz 0 bearbeitet das gesamte Grid eine Aufgabe,
 also rank/count wie beim klassischen Grid-Stride-Loop. Ab Ansatz 1 teilen
 sich mehrere Aufgaben ein Grid, und nur so bleibt die Gesamtarbeit pro
 Aufgabe (num_elems Elemente) über alle Phasen konstant.
*/


__device__ __forceinline__ void compute_ci_fma(float* data, size_t num_elems, size_t iterations,
                                               size_t rank, size_t count) {
    for (size_t thid = rank; thid < num_elems; thid += count) {
        float val0 = data[thid];
        float val1 = val0 + 0.1f;
        float val2 = val0 + 0.2f;
        float val3 = val0 + 0.3f;

        for (size_t i = 0; i < iterations; i++) {
            val0 = fmaf(val0, 1.00001f, 0.00001f);
            val1 = fmaf(val1, 1.00001f, 0.00001f);
            val2 = fmaf(val2, 1.00001f, 0.00001f);
            val3 = fmaf(val3, 1.00001f, 0.00001f);
        }
        data[thid] = val0 + val1 + val2 + val3;
    }
}

__device__ __forceinline__ void compute_mi_indirect(const float* in, float* out,
                                                    const size_t* indices,
                                                    size_t n, size_t iterations,
                                                    size_t rank, size_t count) {
    for (size_t thid = rank; thid < n; thid += count) {
        float sum = 0.0f;
        size_t current_idx = thid;

        for (size_t i = 0; i < iterations; i++) {
            current_idx = indices[current_idx];
            sum += in[current_idx];
        }
        out[thid] = sum;
    }
}

// Unveränderte Schnittstlle für baseline_0/1 und approach_0

__global__ void ci_kernel(float* data, size_t num_elems, size_t iterations) {
    compute_ci_fma(data, num_elems, iterations,
                   blockIdx.x * blockDim.x + threadIdx.x,
                   gridDim.x * blockDim.x);
}

__global__ void mi_kernel(const float* in, float* out, const size_t* indices,
                          size_t n, size_t iterations) {
    compute_mi_indirect(in, out, indices, n, iterations,
                        blockIdx.x * blockDim.x + threadIdx.x,
                        gridDim.x * blockDim.x);
}

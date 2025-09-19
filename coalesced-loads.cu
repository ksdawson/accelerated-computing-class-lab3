// Memory Coalescing Benchmark Kernels
//
// This file contains CUDA kernels to demonstrate and benchmark the performance
// difference between coalesced and non-coalesced memory access patterns:
// - Non-coalesced memory access pattern
// - Coalesced memory access pattern

#include <cuda_runtime.h>
#include <iostream>

using data_type = float;

#define THREADS_PER_WARP 32
#define WARPS 32 // total threads = 32*32 = 1024
#define X 1024   // elements per thread

////////////////////////////////////////////////////////////////////////////////
// Non-Coalesced Memory Access Pattern

__global__ void non_coalesced_load(data_type *dst, data_type *src, int x) {

    // <!-- TODO: your code here -->
}

////////////////////////////////////////////////////////////////////////////////
// Coalesced Memory Access Pattern

__global__ void coalesced_load(data_type *dst, data_type *src, int x) {

    // <!-- TODO: your code here -->
}

////////////////////////////////////////////////////////////////////////////////
///          YOU DO NOT NEED TO MODIFY THE CODE BELOW HERE.                  ///
////////////////////////////////////////////////////////////////////////////////

// CUDA error checking macro
#define CUDA_CHECK(x) \
    do { \
        cudaError_t err = (x); \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error " << static_cast<int>(err) << " (" \
                      << cudaGetErrorString(err) << ") at " << __FILE__ << ":" \
                      << __LINE__ << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)

// Macro to benchmark kernel and print timing results
#define benchmark_and_run(kernel_name, time_var) \
    do { \
        cudaEvent_t start, stop; \
        CUDA_CHECK(cudaEventCreate(&start)); \
        CUDA_CHECK(cudaEventCreate(&stop)); \
        CUDA_CHECK(cudaEventRecord(start)); \
        kernel_name<<<WARPS, THREADS_PER_WARP>>>(d_dst, d_src, X); \
        CUDA_CHECK(cudaEventRecord(stop)); \
        CUDA_CHECK(cudaEventSynchronize(stop)); \
        CUDA_CHECK(cudaEventElapsedTime(&time_var, start, stop)); \
        std::cout << #kernel_name " time = \t" << time_var << " ms" << std::endl; \
        CUDA_CHECK(cudaEventDestroy(start)); \
        CUDA_CHECK(cudaEventDestroy(stop)); \
    } while (0)

int main() {
    // Initialize data and allocate memory
    int total_threads = THREADS_PER_WARP * WARPS;
    int total_elements = total_threads * X;

    data_type *h_src = new data_type[total_elements];
    data_type *h_dst = new data_type[total_elements];
    for (int i = 0; i < total_elements; i++) {
        h_src[i] = static_cast<data_type>(i);
    }

    data_type *d_src = nullptr;
    data_type *d_dst = nullptr;
    CUDA_CHECK(cudaMalloc(&d_src, total_elements * sizeof(data_type)));
    CUDA_CHECK(cudaMalloc(&d_dst, total_elements * sizeof(data_type)));
    CUDA_CHECK(cudaMemcpy(
        d_src,
        h_src,
        total_elements * sizeof(data_type),
        cudaMemcpyHostToDevice));

    // Run benchmarks
    float ms_non, ms_co;
    benchmark_and_run(non_coalesced_load, ms_non);
    benchmark_and_run(coalesced_load, ms_co);

    // Print speedup comparison
    std::cout << "Speedup: " << (ms_non / ms_co) << "x" << std::endl;

    // Clean up memory
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));
    delete[] h_src;
    delete[] h_dst;
}

#include <stdio.h>
#include <cuda_runtime.h>

//#define SIZE 1024*1024*1024*20  // Define the size of the vectors
// Error checking macro
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define gpuKernelCheck() { gpuKernelAssert(__FILE__, __LINE__); }
inline void gpuKernelAssert(const char *file, int line, bool abort=true) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s %s %d\n", cudaGetErrorString(err), file, line);
        if (abort) exit(err);
    }
}

// CUDA Kernel for vector addition
__global__ void vectorAdd(int *A, int *B, int *C, int n) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    int *A, *B, *C;            // Host vectors
    int *d_A, *d_B, *d_C;      // Device vectors
    long long SIZE=1024LL*1024*32;
    long size = SIZE * sizeof(int);

    // CUDA event creation, used for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaError_t err ;
    
    // Allocate device vectors
    cudaCheckError(cudaMalloc((void **)&d_A, size));
    


    err=cudaMalloc((void **)&d_B, size);
    if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device memory - %s\n", cudaGetErrorString(err));
    }



    err=cudaMalloc((void **)&d_C, size);
    if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device memory - %s\n", cudaGetErrorString(err));
    }

// Allocate and initialize host vectors
    A = (int *)malloc(size);
    B = (int *)malloc(size);
    C = (int *)malloc(size);
    for (int i = 0; i < SIZE; i++) {
        A[i] = i;
        B[i] = SIZE - i;
    }


    // Copy host vectors to device
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // Start recording
    cudaEventRecord(start);

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 512;
    int blocksPerGrid = (SIZE + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, SIZE);
    gpuKernelCheck();

    int maxActiveBlocks;
    // This calculates the maximum blocks per SM for your specific kernel
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxActiveBlocks, 
        vectorAdd,       // Name of your __global__ kernel function
        threadsPerBlock,             // The block size you chose (threads per block)
        0                // Dynamic shared memory usage in bytes (0 if none)
    );

    printf("For this kernel, each SM can hold %d blocks simultaneously.\n", maxActiveBlocks);
    /*
    Note Max_threads_per_SM  0: 1536 so 1536 = maxActiveBlocks * threadsPerBlock, gives 100% occupancy (ideal case).

    Every block you create requires a tiny bit of hardware management overhead from the GPU scheduler.
    Managing 16 blocks per SM takes more scheduling effort than managing 6 blocks.
    By using 256 threads, you make the scheduler's job easier.

    Occupancy is just a metric of how many threads are present, not how fast they are executing.
    While 96 threads gave you an $84\%$ Achieved Occupancy vs. $80\%$ for 512 threads,
    the 96-thread version has to manage over 5 times as many blocks total across the grid.
    The hardware overhead of creating, scheduling, and destroying all those extra blocks can
    sometimes completely wipe out the minor $4\%$ gain in occupancy.
    */

    // Stop recording
    cudaEventRecord(stop);

    // Copy result back to host
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    // Calculate and print the execution time
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Execution time: %f milliseconds\n", milliseconds);

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(A);
    free(B);
    free(C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

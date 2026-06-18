#include <stdio.h>
#include <cuda_runtime.h>

#define SIZE 1024*1024*1024  // Define the size of the vectors
// For the size above you may not have enough memory for 3 vectors. 
// Each vector will require 4GB of memory. We use chunk to divide each vector into smaller pieces
// and process them one by one. 
#define CHUNK_SIZE 1024*1024*128  // Define the size of each chunk. 
// So we have 8 chunks in total for each vector.
// Each chunk will require 512MB of memory, which is more manageable for most GPUs.


// CUDA Kernel for vector addition
__global__ void vectorAdd(int *A, int *B, int *C, int chunk_size) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < chunk_size) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    int *chunk_A, *chunk_B, *chunk_C;            // Host vectors
    int *d_A, *d_B, *d_C;      // Device vectors
    size_t size = CHUNK_SIZE * sizeof(int);


    // Allocate host vectors.
    //(If you move these inside the loop, we will allocate and free memory for each chunk, which is inefficient. You can try)
    chunk_A = (int *)malloc(size);
    chunk_B = (int *)malloc(size);
    chunk_C = (int *)malloc(size);

    // Allocate device vectors. You sould have GPU with 1.5GB of memory to run this code.
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    int threadsPerBlock = 128;
    int blocksPerGrid = (CHUNK_SIZE + threadsPerBlock - 1) / threadsPerBlock;

    for (int chunk = 0; chunk < SIZE; chunk += CHUNK_SIZE) {
       // CUDA event creation, used for timing
       cudaEvent_t start, stop;
       cudaEventCreate(&start);
       cudaEventCreate(&stop);

      for (int i = 0; i < CHUNK_SIZE; i++) {
          chunk_A[i] = i;
          chunk_B[i] = CHUNK_SIZE - i;
      }

      // Copy host vectors to device
      cudaMemcpy(d_A, chunk_A, size, cudaMemcpyHostToDevice);
      cudaMemcpy(d_B, chunk_B, size, cudaMemcpyHostToDevice);

      // Start recording
      cudaEventRecord(start);

      // Launch the Vector Add CUDA Kernel
      vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, CHUNK_SIZE);
      cudaError_t err = cudaGetLastError();
      printf("%s\n", cudaGetErrorString(err));

      // Stop recording
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);

      // Copy result back to host
      cudaMemcpy(chunk_C, d_C, size, cudaMemcpyDeviceToHost);

      // Calculate and print the execution time
      float milliseconds = 0;
      cudaEventElapsedTime(&milliseconds, start, stop);
      printf("Execution time with num_blocks = %d and threads_per_block = %d: %f milliseconds\n", blocksPerGrid, threadsPerBlock, milliseconds);


      cudaEventDestroy(start);
      cudaEventDestroy(stop);
    }
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(chunk_A);
    free(chunk_B);
    free(chunk_C);

    return 0;
}

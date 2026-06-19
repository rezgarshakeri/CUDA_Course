#include <stdio.h>

int main() {
    int nDevices;

    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        // Fetch the memory clock rate using the modern attribute API
        int memoryClockRateKHz = 0;
        cudaDeviceGetAttribute(&memoryClockRateKHz, cudaDevAttrMemoryClockRate, i);

        printf("  Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (KHz): %d\n", memoryClockRateKHz);
        printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
               2.0 * memoryClockRateKHz * (prop.memoryBusWidth / 8) / 1.0e6);
        printf("  Total global memory: %lu\n", prop.totalGlobalMem);
        printf("  Compute capability: %d.%d\n",
               prop.major, prop.minor);
        printf("  Number of SMs: %d\n",
               prop.multiProcessorCount);
        printf("  Max threads per block: %d\n",
               prop.maxThreadsPerBlock);
        printf("  Max threads dimensions: x = %d, y = %d, z = %d\n",
               prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  Max grid dimensions: x = %d, y = %d, z = %d\n",
               prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    }
    return 0;
}
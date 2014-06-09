#include "helper_cuda.h"

#include "utils.cuh"
#include "knn_gpgpu.h"

void cuSetDevice(int device)
{
    checkCudaErrors(cudaSetDevice(device));
}

int cuGetDevice()
{
    int device;
    checkCudaErrors(cudaGetDevice(&device));
    return device;
}

int cuGetDeviceCount()
{
    int device_count;
    checkCudaErrors(cudaGetDeviceCount(&device_count));
    return device_count;
}

size_t getFreeBytesOnGpu()
{
    size_t free_byte, total_byte ;
    cudaError_t cuda_status = cudaMemGetInfo( &free_byte, &total_byte ) ;
    return free_byte - 1024;
}

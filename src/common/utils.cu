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

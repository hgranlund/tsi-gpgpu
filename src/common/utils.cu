#include "helper_cuda.h"

#include "utils.cuh"
#include "knn_gpgpu.h"


void cuSetDevice(int devive)
{
    checkCudaErrors(cudaSetDevice(devive));
}
void cuGetDevice(int *devive)
{
    checkCudaErrors(cudaGetDevice(devive));
}

void cuGetDeviceCount(int *device_count)
{
    checkCudaErrors(cudaGetDeviceCount(device_count));
}

void cuStreamCreate(cudaStream_t *pStream, int device)
{
    int device_orig;
    cuGetDevice(&device_orig);
    cuSetDevice(device);
    checkCudaErrors(cudaStreamCreate(pStream));
    cuSetDevice(device_orig);
}

void cuStreamSynchronize(cudaStream_t stream)
{
    checkCudaErrors(cudaStreamSynchronize(stream));
}

void cuStreamDestroy(cudaStream_t stream)
{
    checkCudaErrors(cudaStreamDestroy(stream));
}

void cuGetDeviceProperties(struct cudaDeviceProp *prop, int device)
{
    cudaGetDeviceProperties(prop, device);
}


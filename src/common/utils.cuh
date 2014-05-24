#ifndef _CU_UTILS_
#define _CU_UTILS_
#include <cuda_runtime.h>

void cuSetDevice(int devive);
void cuGetDevice(int *devive);
void cuGetDeviceCount(int *device_count);
void cuStreamCreate(cudaStream_t *pStream, int device);
void cuStreamSynchronize(cudaStream_t stream);
void cuStreamDestroy(cudaStream_t stream);
void cuGetDeviceProperties(struct cudaDeviceProp *prop, int device);

#endif

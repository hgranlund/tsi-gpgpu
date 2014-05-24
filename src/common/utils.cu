#include "helper_cuda.h"

#include "utils.cuh"
#include "knn_gpgpu.h"



void cuSetDevice(int devive)
{
    checkCudaErrors(cudaSetDevice(devive));
}



#include "kNN-brute-force-bitonic.cuh"
// #include "../kernels/reduction.cuh"
#include "../kernels/reduction-mod.cuh"

#include <stdio.h>
#include <math.h>

#include "helper_cuda.h"



// #define SHARED_SIZE_LIMIT 1024U
#define SHARED_SIZE_LIMIT 512U
#define checkCudaErrors(val)           check ( (val), #val, __FILE__, __LINE__ )

__constant__  float d_query[3];

__global__ void cuComputeDistance( float *ref, unsigned int ref_nb , unsigned int dim,  Distance *dist)
{

    float dx, dy, dz;

    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    while (index < ref_nb)
    {
        dx = ref[index * dim] - d_query[0];
        dy = ref[index * dim + 1] - d_query[1];
        dz = ref[index * dim + 2] - d_query[2];
        dist[index].value = pow(dx, 2) + pow(dy, 2) + pow(dz, 2);
        dist[index].index = index;
        index += gridDim.x * blockDim.x;
    }
}
__global__ void cuParallelSqrt(Distance *dist, unsigned int k)
{
    unsigned int xIndex = blockIdx.x;
    if (xIndex < k)
    {
        dist[xIndex].value = rsqrt(dist[xIndex].value);
    }
}


void knn_brute_force_reduce(float *h_ref, unsigned int ref_nb, float *h_query, unsigned int dim, unsigned int k, float *dist, int *ind)
{

    float        *d_ref;
    Distance     *d_dist, *h_dist;
    unsigned int i;
    h_dist = (Distance *) malloc(k * sizeof(Distance));
    checkCudaErrors(cudaMalloc( (void **) &d_dist, ref_nb * sizeof(Distance)));
    checkCudaErrors(cudaMalloc( (void **) &d_ref, ref_nb * sizeof(float) * dim));

    checkCudaErrors(cudaMemcpy(d_ref, h_ref, ref_nb * dim * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(d_query, h_query, dim * sizeof(float)));

    unsigned int threadCount = min(ref_nb, SHARED_SIZE_LIMIT);
    unsigned int blockCount = ref_nb / threadCount;
    blockCount = min(blockCount, 65536);
    cuComputeDistance <<< blockCount, threadCount>>>(d_ref, ref_nb, dim, d_dist);
    for (i = 0; i < k; ++i)
    {
        dist_min_reduce(d_dist + i, ref_nb - i);
    }
    cuParallelSqrt <<< k, 1>>>(d_dist, k);
    checkCudaErrors(cudaMemcpy(h_dist, d_dist, k * sizeof(Distance), cudaMemcpyDeviceToHost));
    for (i = 0; i < k; ++i)
    {
        dist[i] = h_dist[i].value;
        ind[i] = h_dist[i].index;
    }
    checkCudaErrors(cudaFree(d_ref));
    checkCudaErrors(cudaFree(d_dist));
}




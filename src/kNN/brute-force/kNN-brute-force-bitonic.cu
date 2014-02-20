

// Includes
#include <kNN-brute-force-bitonic.cuh>
#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include <time.h>
#include <assert.h>

#include "helper_cuda.h"

// #define SHARED_SIZE_LIMIT 1024U
#define SHARED_SIZE_LIMIT 512U
#define checkCudaErrors(val)           check ( (val), #val, __FILE__, __LINE__ )

__device__ void cuCompare_b(float &distA, int &indA, float &distB, int &indB, int dir)
{
  float f;
  int i;
  if ((distA  >= distB) == dir)
  {
    f = distA;
    distA  = distB;
    distB = f;
    i = indA;
    indA = indB;
    indB = i;
  }
}
__constant__  float query_dev[3];

__global__ void cuComputeDistanceGlobal( float* ref, int ref_nb , int dim,  float* dist, int* ind){

  float dx,dy,dz;

  int index = blockIdx.x*blockDim.x+threadIdx.x;
  while (index < ref_nb){
    dx=ref[index*dim] - query_dev[0];
    dy=ref[index*dim + 1] - query_dev[1];
    dz=ref[index*dim + 2] - query_dev[2];
    dist[index] = (dx*dx)+(dy*dy)+(dz*dz);
    ind[index] = index;
    index += gridDim.x*blockDim.x;
  }
}

__global__ void cuBitonicSortOneBlock(float* dist, int* ind, int n,int dir){

  int blockoffset = blockIdx.x * blockDim.x *2;
  dist+=blockoffset;
  ind+=blockoffset;

  for (int size = 2; size <= blockDim.x*2; size <<= 1)
  {
    int ddd = dir ^ ((threadIdx.x & (size / 2)) != 0);
    for (int stride = size / 2; stride > 0; stride >>= 1)
    {
      __syncthreads();
      int pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
      cuCompare_b(dist[pos], ind[pos], dist[pos + stride], ind[pos + stride],ddd);
    }
  }
}
__global__ void cuBitonicSort(float* dist, int* ind, int n,int dir){

  int blockoffset = blockIdx.x * blockDim.x *2;
  dist+=blockoffset;
  ind+=blockoffset;

  for (int size = 2; size <= blockDim.x*2; size <<= 1)
  {
    int ddd = dir ^ ((threadIdx.x & (size / 2)) != 0);
    for (int stride = size / 2; stride > 0; stride >>= 1)
    {
      __syncthreads();
      int pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
      cuCompare_b(dist[pos], ind[pos], dist[pos + stride], ind[pos + stride],ddd);
    }
  }

  int ddd = blockIdx.x & 1;
  {
    for (int stride = blockDim.x; stride > 0; stride >>= 1)
    {
      __syncthreads();
      int pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
      cuCompare_b(dist[pos], ind[pos], dist[pos + stride], ind[pos + stride],ddd);
    }
  }
}

__global__ void cuBitonicMergeGlobal(float* dist, int* ind, int n, int size, int stride ,int dir)
{
  int global_comparatorI = blockIdx.x * blockDim.x + threadIdx.x;
  int        comparatorI = global_comparatorI & (n / 2 - 1);

  int ddd = dir ^ ((comparatorI & (size / 2)) != 0);
  int pos = 2 * global_comparatorI - (global_comparatorI & (stride - 1));
  cuCompare_b(dist[pos], ind[pos], dist[pos + stride], ind[pos + stride],ddd);
}


__global__ void cuParallelSqrt_b(float *dist, int k){
  unsigned int xIndex = blockIdx.x;
  if (xIndex < k){
    dist[xIndex] = sqrt(dist[xIndex]);
  }
}


__global__ void cuBitonicMergeShared(float* dist, int* ind, int n, int size, int dir)
{

  int blockoffset = blockIdx.x * blockDim.x *2;
  dist+=blockoffset;
  ind+=blockoffset;
  int comparatorI = (blockIdx.x * blockDim.x + threadIdx.x) & ((n / 2) - 1);
  int ddd = dir ^ ((comparatorI & (size / 2)) != 0);
  for (int stride = blockDim.x; stride > 0; stride >>= 1)
  {
    __syncthreads();
    int pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
    cuCompare_b(dist[pos], ind[pos], dist[pos + stride], ind[pos + stride],ddd);
  }
}

void bitonic_sort(float* dist_dev, int* ind_dev, int n, int dir){
  int max_threads_per_block = min(SHARED_SIZE_LIMIT, n);
  int  blockCount = n / max_threads_per_block;
  int threadCount = max_threads_per_block/ 2;
  blockCount = max(1, blockCount);
  threadCount = min(max_threads_per_block, threadCount);
  if (blockCount==1){
    cuBitonicSortOneBlock<<<blockCount, threadCount>>>(dist_dev,ind_dev, n, dir);
  }
  else{
    cuBitonicSort<<<blockCount, threadCount>>>(dist_dev,ind_dev, n, dir);
    for (int size = 2 * max_threads_per_block; size <= n; size <<= 1){
      for (int stride = size / 2; stride > 0; stride >>= 1){
        if (stride >= max_threads_per_block)
        {
          cuBitonicMergeGlobal<<<blockCount, threadCount>>>(dist_dev,ind_dev, n, size, stride, dir);
        }
        else
        {
          cuBitonicMergeShared<<<blockCount, threadCount>>>(dist_dev,ind_dev, n, size, dir);
          break;
        }
      }
    }
  }
}

int factorRadix2(int *log2L, int L)
{
  if (!L)
  {
    *log2L = 0;
    return 0;
  }
  else
  {
    for (*log2L = 0; (L & 1) == 0; L >>= 1, *log2L++);
      return L;
  }
}

void knn_brute_force_bitonic(float* ref_host, int ref_nb, float* query_host, int dim, int k, float* dist_host, int* ind_host){

  unsigned int size_of_float = sizeof(float);
  unsigned int size_of_int   = sizeof(int);

  float        *ref_dev;
  float        *dist_dev;
  int          *ind_dev;


  int log2L;
  int factorizationRemainder = factorRadix2(&log2L, ref_nb);
  // assert(factorizationRemainder == 1);

  checkCudaErrors(cudaMalloc( (void **) &dist_dev, ref_nb * size_of_float));
  checkCudaErrors(cudaMalloc( (void **) &ind_dev, ref_nb * size_of_int));
  checkCudaErrors(cudaMalloc( (void **) &ref_dev, ref_nb * size_of_float * dim));

  checkCudaErrors(cudaMemcpy(ref_dev, ref_host, ref_nb*dim*size_of_float, cudaMemcpyHostToDevice));


  checkCudaErrors(cudaMemcpyToSymbol(query_dev, query_host, dim*size_of_float));
  int threadCount = min(ref_nb, SHARED_SIZE_LIMIT);
  int blockCount = ref_nb/threadCount;
  blockCount = min(blockCount, 65536);
  cuComputeDistanceGlobal<<<blockCount,threadCount>>>(ref_dev, ref_nb, dim, dist_dev, ind_dev);
  bitonic_sort(dist_dev,ind_dev, ref_nb, 1);
  cuParallelSqrt_b<<<k,1>>>(dist_dev, k);

  checkCudaErrors(cudaMemcpy(dist_host, dist_dev, k*size_of_float, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(ind_host,  ind_dev,  k*size_of_int, cudaMemcpyDeviceToHost));

  checkCudaErrors(cudaFree(ref_dev));
  checkCudaErrors(cudaFree(dist_dev));
  checkCudaErrors(cudaFree(ind_dev));
}




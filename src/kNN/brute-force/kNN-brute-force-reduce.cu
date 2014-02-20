

// Includes
#include <kNN-brute-force-bitonic.cuh>
#include "../kernels/reduction.cuh"

#include <stdio.h>
#include <math.h>
#include <cuda.h>

#include "helper_cuda.h"


// #define SHARED_SIZE_LIMIT 1024U
#define SHARED_SIZE_LIMIT 512U
// #define SHARED_SIZE_LIMIT 5012U
#define checkCudaErrors(val)           check ( (val), #val, __FILE__, __LINE__ )

    __device__ void cuCompare(float &distA, int &indA, float &distB, int &indB, int dir)
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

    __global__ void cuComputeDistance( float* ref, int ref_nb , int dim,  float* dist, int* ind){

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
    __global__ void cuParallelSqrt(float *dist, int k){
      unsigned int xIndex = blockIdx.x;
      if (xIndex < k){
        dist[xIndex] = sqrt(dist[xIndex]);
      }
    }


    void knn_brute_force_reduce(float* ref_host, int ref_nb, float* query_host, int dim, int k, float* dist_host, int* ind_host){

      unsigned int size_of_float = sizeof(float);
      unsigned int size_of_int   = sizeof(int);

      float        *ref_dev;
      float        *dist_dev;
      int          *ind_dev;


      checkCudaErrors(cudaMalloc( (void **) &dist_dev, ref_nb * size_of_float));
      checkCudaErrors(cudaMalloc( (void **) &ind_dev, ref_nb * size_of_int));
      checkCudaErrors(cudaMalloc( (void **) &ref_dev, ref_nb * size_of_float * dim));

      checkCudaErrors(cudaMemcpy(ref_dev, ref_host, ref_nb*dim*size_of_float, cudaMemcpyHostToDevice));
      checkCudaErrors(cudaMemcpyToSymbol(query_dev, query_host, dim*size_of_float));

      int threadCount = min(ref_nb, SHARED_SIZE_LIMIT);
      int blockCount = ref_nb/threadCount;
      blockCount = min(blockCount, 65536);
      cuComputeDistance<<<blockCount,threadCount>>>(ref_dev, ref_nb, dim, dist_dev, ind_dev);
      for (int i = 0; i < k; ++i)
      {
        knn_min_reduce(dist_dev+i, ind_dev+i, ref_nb-i);
      }
      cuParallelSqrt<<<k,1>>>(dist_dev, k);
      checkCudaErrors(cudaMemcpy(ind_host,  ind_dev,  k*size_of_int, cudaMemcpyDeviceToHost));
      checkCudaErrors(cudaMemcpy(dist_host, dist_dev, k*size_of_float, cudaMemcpyDeviceToHost));
      checkCudaErrors(cudaFree(ref_dev));
      checkCudaErrors(cudaFree(dist_dev));
      checkCudaErrors(cudaFree(ind_dev));
    }




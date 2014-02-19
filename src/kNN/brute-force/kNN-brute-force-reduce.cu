

// Includes
#include <kNN-brute-force-bitonic.cuh>
#include "../kernels/reduction.cuh"

#include <stdio.h>
#include <math.h>
#include <cuda.h>

#include "helper_cuda.h"



// #define SHARED_SIZE_LIMIT 1024U
#define SHARED_SIZE_LIMIT 512U
#define checkCudaErrors(val)           check ( (val), #val, __FILE__, __LINE__ )

    __device__ void cuCompare(Distance &distA,  Distance &distB, int dir)
    {
      Distance f;
      if ((distA.value  >= distB.value) == dir)
      {
        f = distA;
        distA  = distB;
        distB = f;
      }
    }
    __constant__  float d_query[3];

    __global__ void cuComputeDistance( float* ref, int ref_nb , int dim,  Distance* dist){

      float dx,dy,dz;

      int index = blockIdx.x*blockDim.x+threadIdx.x;
      while (index < ref_nb){
        dx=ref[index*dim] - d_query[0];
        dy=ref[index*dim + 1] - d_query[1];
        dz=ref[index*dim + 2] - d_query[2];
        dist[index].value = (dx*dx)+(dy*dy)+(dz*dz);
        dist[index].index = index;
        index += gridDim.x*blockDim.x;
      }
    }
    __global__ void cuParallelSqrt(Distance *dist, int k){
      unsigned int xIndex = blockIdx.x;
      if (xIndex < k){
        dist[xIndex].value = sqrt(dist[xIndex].value);
      }
    }


    void knn_brute_force_reduce(float* h_ref, int ref_nb, float* h_query, int dim, int k, Distance* h_dist){

      float        *d_ref;
      Distance        *d_dist;


      checkCudaErrors(cudaMalloc( (void **) &d_dist, ref_nb * sizeof(Distance)));
      checkCudaErrors(cudaMalloc( (void **) &d_ref, ref_nb * sizeof(float) * dim));

      checkCudaErrors(cudaMemcpy(d_ref, h_ref, ref_nb*dim*sizeof(float), cudaMemcpyHostToDevice));
      checkCudaErrors(cudaMemcpyToSymbol(d_query, h_query, dim*sizeof(float)));

      int threadCount = min(ref_nb, SHARED_SIZE_LIMIT);
      int blockCount = ref_nb/threadCount;
      blockCount = min(blockCount, 65536);
      cuComputeDistance<<<blockCount,threadCount>>>(d_ref, ref_nb, dim, d_dist);
      for (int i = 0; i < k; ++i)
      {
        knn_min_reduce(d_dist+i, ref_nb-i);
      }
      cuParallelSqrt<<<k,1>>>(d_dist, k);
      checkCudaErrors(cudaMemcpy(h_dist, d_dist, k*sizeof(Distance), cudaMemcpyDeviceToHost));
      checkCudaErrors(cudaFree(d_ref));
      checkCudaErrors(cudaFree(d_dist));
    }




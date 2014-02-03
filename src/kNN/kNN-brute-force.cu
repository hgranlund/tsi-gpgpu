

// Includes
#include <kNN-brute-force.cuh>
#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include <time.h>
#include "bitonic-sort-serial.c"

__constant__  float query_dev[3];

__global__ void cuComputeDistanceGlobal( float* ref, int ref_nb , int dim,  float* dist, int* ind){

  // restiction: dim=3
  float dx,dy,dz;

  int index = blockIdx.x * dim;
  while (index < ref_nb){
    dx=ref[index] - query_dev[0];
    dy=ref[index + 1] - query_dev[1];
    dz=ref[index + 2] - query_dev[2];
    dist[index/dim] = (dx*dx)+(dy*dy)+(dz*dz);
    ind[index/dim] = index/dim;
    index += gridDim.x * dim;
  }
}


__global__ void cuParallelSqrt(float *dist, int k){
  unsigned int xIndex = blockIdx.x;
  if (xIndex < k){
    dist[xIndex] = sqrt(dist[xIndex]);
  }
}



void knn_brute_force(float* ref_host, int ref_nb, float* query_host, int dim, int k, float* dist_host, int* ind_host){

  unsigned int size_of_float = sizeof(float);
  unsigned int size_of_int   = sizeof(int);

  float        *ref_dev;
  float        *dist_dev;
  int          *ind_dev;


  cudaMalloc( (void **) &dist_dev, ref_nb * size_of_float);
  cudaMalloc( (void **) &ind_dev, ref_nb * size_of_int);
  cudaMalloc( (void **) &ref_dev, ref_nb * size_of_float * dim);


  cudaMemcpy(ref_dev, ref_host, ref_nb*dim*size_of_float, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(query_dev, query_host, dim*size_of_float);

  cuComputeDistanceGlobal<<<256,1>>>(ref_dev, ref_nb, dim, dist_dev, ind_dev);

  // cuParallelSqrt<<<k,1>>>(dist_dev, k);
  cudaMemcpy(dist_host, dist_dev, k*size_of_float, cudaMemcpyDeviceToHost);
  cudaMemcpy(ind_host,  ind_dev,  k*size_of_int, cudaMemcpyDeviceToHost);

  printf("\n[");
  for (int i = 0; i < k; ++i)
  {
    printf("%d, ",ind_host[i] );
  }
  printf("]\n");


  printf("\n[");
  for (int i = 0; i < k; ++i)
  {
    printf("%f, ",dist_host[i]);
  }
  printf("]\n");




  // bitonic_sort_serial(dist_host, k);

  cudaFree(ref_dev);
  cudaFree(ind_dev);
  cudaFree(query_dev);
}



// Includes
#include <kNN-brute-force-reduce.cuh>
#include <knn_gpgpu.h>
#include "gtest/gtest.h"

#include <stdio.h>
#include <cuda.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#define checkCudaErrors(val)           check ( (val), #val, __FILE__, __LINE__ )


TEST(knn_brute_force, test_knn_reduce_correctness){

  float *ref, *dist;
  float *query;
  int *ind;
  unsigned int    ref_nb = 131072;
  unsigned int    query_nb = 1;
  unsigned int    dim=3;
  unsigned int    k          = 100;
  unsigned int    iterations = 1;
  unsigned int    i;

  ref    = (float *) malloc(ref_nb   * dim * sizeof(float));
  query  = (float *) malloc(query_nb * dim * sizeof(float));
  dist  = (float *) malloc(k * sizeof(float));
  ind  = (int *) malloc(k * sizeof(float));

  for (unsigned int count = 0; count < ref_nb*dim; count++)
  {
    ref[count] = (float)ref_nb*dim-count;
  }
  for (unsigned int count = 0; count < query_nb*dim; count++)
  {
    query[count] = 0;
  }

  for (i=0; i<iterations; i++){
    knn_brute_force_reduce(ref, ref_nb, query, dim, k, dist, ind);
  }

  for (unsigned int i = 0; i < k; ++i)
  {
    ASSERT_EQ(ind[i], ref_nb-1-i) << "Faild with i = "<<i << " and n = " << ref_nb;;
  }

  free(dist);
  free(ind);
  free(query);
  free(ref);
  cudaDeviceSynchronize();
  cudaDeviceReset();
}

TEST(knn_brute_force, test_knn_reduce_time){

  float *ref, *dist;
  float *query;
  int *ind;
  unsigned int    ref_nb = 8388608;
  unsigned int    query_nb = 1;
  unsigned int    dim=3;
  unsigned int    k          = 100;

  ref    = (float *) malloc(ref_nb   * dim * sizeof(float));
  query  = (float *) malloc(query_nb * dim * sizeof(float));
  dist  = (float *) malloc(k * sizeof(float));
  ind  = (int *) malloc(k * sizeof(float));

  for (unsigned int count = 0; count < ref_nb*dim; count++)
  {
    ref[count] = (float)ref_nb*dim-count;
  }
  for (unsigned int count = 0; count < query_nb*dim; count++)
  {
    query[count] = 0;
  }

  cudaEvent_t start, stop;
  unsigned int bytes = ref_nb * (sizeof(float));
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));
  float elapsed_time=0;

  checkCudaErrors(cudaEventRecord(start, 0));

  knn_brute_force_reduce(ref, ref_nb, query, dim, k, dist, ind);


  checkCudaErrors(cudaEventRecord(stop, 0));
  cudaEventSynchronize(start);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_time, start, stop);
  double throughput = 1.0e-9 * ((double)bytes)/(elapsed_time* 1e-3);
  printf("kNN-brute-force-reduce, Throughput = %.4f GB/s, Time = %.5f ms, Size = %u Elements, k = %u NumDevsUsed = %d\n",
   throughput, elapsed_time, ref_nb,k, 1);


  for (unsigned int i = 0; i < k; ++i)
  {
    ASSERT_EQ(ind[i], ref_nb-1-i) << "Faild with i = "<<i << " and n = " << ref_nb;;
  }

  free(dist);
  free(ind);
  free(query);
  free(ref);
  cudaDeviceReset();
  cudaDeviceSynchronize();
}




// Includes
#include <kNN-brute-force.cuh>
#include <knn_gpgpu.h>
#include <stdio.h>
#include "gtest/gtest.h"

#include <math.h>
#include <cuda.h>
#include <time.h>
#include <assert.h>
#include "../../../common/common-debug.c"



TEST(knn_brute_force, test_knn_brute_force_give_rigth_result_with_6553_points){
 // Variables and parameters
  float* ref;                 // Pointer to reference point array
  float* query;               // Pointer to query point array
  float* dist;                // Pointer to distance array
  int*   ind;                 // Pointer to index array
  int    ref_nb     = 8;   // Reference point number, max=65535
  int    query_nb   = 1;   // Query point number,     max=65535
  int    dim        = 3;     // Dimension of points
  int    k          = 8;     // Nearest neighbors to consider
  int    i;

  // Memory allocation
  ref    = (float *) malloc(ref_nb   * dim * sizeof(float));
  query  = (float *) malloc(query_nb * dim * sizeof(float));
  dist   = (float *) malloc( k * sizeof(float));
  ind    = (int *)   malloc( k * sizeof(int));

  // Init
  srand(time(NULL));
  for (i=0 ; i<ref_nb   * dim ; i++)
  {
    ref[i]    = (float)rand() / (float)RAND_MAX;
  }
  for (i=0 ; i<query_nb * dim ; i++)
  {
    query[i]  = (float)rand() / (float)RAND_MAX;
  }

  // Variables for duration evaluation
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float elapsed_time;

  // Display informations

  // Call kNN search CUDA
  cudaEventRecord(start, 0);

  knn_brute_force(ref, ref_nb, query, dim, k, dist, ind);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_time, start, stop);
  // printf(" done in %f s\n", elapsed_time/1000);

  // Destroy cuda event object and free memory
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  free(ind);
  free(dist);
  free(query);
  free(ref);
}


TEST(knn_brute_force, test_bitonic_sort){

  float *h_dist, *d_dist;
  int *h_ind, *d_ind;
  int i,n;
  for (n = 8; n < 2097152*2; n <<=1)
  {

    h_dist = (float*) malloc(n*sizeof(float));
    h_ind= (int*) malloc(n*sizeof(int));
    srand(time(NULL));
    for (i=0 ; i<n; i++)
    {
      // h_dist[i]    = n-i-1  ;
      h_dist[i]    = (int)rand();
      h_ind[i]=i;
    }

    cudaMalloc( (void **) &d_dist, n* sizeof(float));
    cudaMalloc( (void **) &d_ind, n * sizeof(int));

    cudaMemcpy(d_dist, h_dist, n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ind, h_ind, n*sizeof(int), cudaMemcpyHostToDevice);
    printArray(h_dist,n);

    bitonic_sort(d_dist,d_ind, n, 1);
    cudaMemcpy(h_dist,d_dist, n*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ind,d_ind , n*sizeof(int), cudaMemcpyDeviceToHost);
    printArray(h_dist,n);

    float last_value = h_dist[0];
    for (i = 0; i < n; ++i)
    {
      ASSERT_LE(last_value, h_dist[i]) << "Faild with i = "<<i << " and n = " << n;
      last_value=h_dist[i];
    }


    free(h_ind);
    free(h_dist);
    cudaFree(d_dist);
    cudaFree(d_ind);

  }
}

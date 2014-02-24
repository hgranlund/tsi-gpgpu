

// Includes
#include <kNN-brute-force-bitonic.cuh>
#include <knn_gpgpu.h>
#include <stdio.h>
#include "gtest/gtest.h"

#include <math.h>
#include <cuda.h>
#include <time.h>
#include <assert.h>
#include "../../../common/common-debug.h"




TEST(knn_brute_force, test_knn_bitonic_correctness){

  float* ref;                 // Pointer to reference point array
  float* query;               // Pointer to query point array
  float* dist;                // Pointer to distance array
  int*   ind;                 // Pointer to index array
  int    ref_nb;              // Reference point number, max=65535
  int    query_nb;            // Query point number,     max=65535
  int    dim;                 // Dimension of points
  int    k          = 10;     // Nearest neighbors to consider
  int    iterations = 1;
  int    i;

  char fileName[] = "data/knn_brute_force_6553_ref_points_1_query_point.data";
  FILE* file = fopen(fileName, "rb");

  fread(&ref_nb, sizeof(int), 1, file);
  ref_nb=4096;
  fread(&query_nb, sizeof(int), 1, file);
  fread(&dim, sizeof(int), 1, file);
  ref    = (float *) malloc(ref_nb   * dim * sizeof(float));
  query  = (float *) malloc(query_nb * dim * sizeof(float));
  dist   = (float *) malloc(query_nb * k * sizeof(float));
  ind    = (int *)   malloc(query_nb * k * sizeof(int));

  for (int count = 0; count < ref_nb*dim; count++)
  {
    fread(&ref[count], sizeof(float), 1, file);
  }
  for (int count = 0; count < query_nb*dim; count++)
  {
    fread(&query[count], sizeof(float), 1, file);
  }

  fclose(file);

  for (i=0; i<iterations; i++){
    knn_brute_force_bitonic(ref, ref_nb, query, dim, k, dist, ind);
  }
  int correct_ind[] = {119, 3309, 3515, 2455, 3172, 1921, 3803, 919, 1048, 244};
  for (int i = 0; i < k; ++i)
  {
    ASSERT_EQ(ind[i], correct_ind[i]);
  }

  free(ind);
  free(dist);
  free(query);
  free(ref);
  cudaDeviceSynchronize();
  cudaDeviceReset();
}


TEST(knn_brute_force, test_bitonic_sort){

  float *h_dist,*h_dist_orig, *d_dist;
  int *h_ind,*h_ind_orig, *d_ind;
  int i,n;
  for (n = 16384; n <=16384; n <<=1)
  {

    h_dist = (float*) malloc(n*sizeof(float));
    h_dist_orig = (float*) malloc(n*sizeof(float));
    h_ind= (int*) malloc(n*sizeof(int));
    h_ind_orig= (int*) malloc(n*sizeof(int));
    srand ( (unsigned int)time(NULL) );
    for (i=0 ; i<n; i++)
    {
      h_dist_orig[i]    = (float)rand();
      h_ind_orig[i]=i;
    }

    cudaMalloc( (void **) &d_dist, n* sizeof(float));
    cudaMalloc( (void **) &d_ind, n * sizeof(int));

    cudaMemcpy(d_dist, h_dist_orig, n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ind, h_ind_orig, n*sizeof(int), cudaMemcpyHostToDevice);

    bitonic_sort(d_dist,d_ind, n, 1);
    cudaMemcpy(h_dist,d_dist, n*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ind,d_ind , n*sizeof(int), cudaMemcpyDeviceToHost);

    float last_value = h_dist[0];
    for (i = 0; i < n; ++i)
    {
      ASSERT_LE(last_value, h_dist[i]) << "Faild with i = "<<i << " and n = " << n;
      last_value=h_dist[i];
    }
    for (i = 0; i < n; ++i)
    {
      ASSERT_LE(h_dist[i], h_dist_orig[h_ind[i]]) << "Faild with i = "<<i << " and n = " << n;
    }

    free(h_ind);
    free(h_dist);
    cudaFree(d_dist);
    cudaFree(d_ind);
    cudaDeviceSynchronize();
    cudaDeviceReset();
  }
}

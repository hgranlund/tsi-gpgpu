

// Includes
#include <kNN-brute-force-reduce.cuh>
#include <knn_gpgpu.h>
#include "gtest/gtest.h"

#include <stdio.h>
#include <cuda.h>


TEST(knn_brute_force, test_knn_reduce_correctness){

  float* ref;
  float* query;
  Distance* dist;
  int    ref_nb = 131072;
  int    query_nb = 1;
  int    dim=3;
  int    k          = 50;
  int    iterations = 1;
  int    i;

  ref    = (float *) malloc(ref_nb   * dim * sizeof(float));
  query  = (float *) malloc(query_nb * dim * sizeof(float));
  dist   = (Distance *) malloc(query_nb * k * sizeof(Distance));

  for (int count = 0; count < ref_nb*dim; count++)
  {
    ref[count] = ref_nb*dim-count;
  }
  for (int count = 0; count < query_nb*dim; count++)
  {
    query[count] = 0;
  }

  for (i=0; i<iterations; i++){
    knn_brute_force_reduce(ref, ref_nb, query, dim, k, dist);
  }

  for (int i = 0; i < k; ++i)
  {
    ASSERT_EQ(dist[i].index, ref_nb-1-i)<< "Faild with i = "<<i << " and n = " << ref_nb;;
  }

  free(dist);
  free(query);
  free(ref);
}


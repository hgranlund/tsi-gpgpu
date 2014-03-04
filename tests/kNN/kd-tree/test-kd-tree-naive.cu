#include <kd-tree-naive.cuh>
#include <knn_gpgpu.h>

#include <stdio.h>
#include <helper_cuda.h>
#include "gtest/gtest.h"


#define debug 0

void print_tree(float *tree, int level, int lower, int upper, int n)
{
  if (debug)
  {
    if (lower >= upper)
    {
      return;
    }

    int i, r = midpoint(lower, upper);

    printf("|");
    for (i = 0; i < level; ++i)
    {
      printf("--");
    }
    printf("(%3.1f, %3.1f, %3.1f)\n", tree[h_index(r, 0, n)], tree[h_index(r, 1, n)], tree[h_index(r, 2, n)]);

    print_tree(tree, 1 + level, lower, r, n);
    print_tree(tree, 1 + level, r + 1, upper, n);
  }
}

TEST(kd_tree_naive, kd_tree_naive_correctness){
  int i, j, n = 1000;
  float *points, *expected_points;
  points = (float*) malloc(n * 3 * sizeof(float));
  expected_points = (float*) malloc(n * 3 * sizeof(float));
  srand(time(NULL));
  for ( i = 0; i < n; ++i)
  {
    for ( j = 0; j < 3; ++j)
    {
      points[h_index(i, j, n)] = n - i -1 +j;
      points[h_index(i, j, n)] = (float) rand() /100000000;
      expected_points[h_index(i, j, n)] = i ;
    }
  }
  if (debug)
  {


    printf("kd tree:\n");
    print_tree(points, 0, 0, n, n);
    printf("==================\n");

  }

  build_kd_tree(points, n);

  for ( i = 0; i < n; ++i)
  {
    for ( j = 0; j < 3; ++j)
    {
// ASSERT_EQ(points[h_index(i, j, 5n)] ,i) << "Faild with i = " << i << " j = " <<j ;
    }
  }

  if (debug)
  {

    printf("kd tree:\n");
    print_tree(points, 0, 0, n, n);
    printf("==================\n");

  }

  free(points);
  free(expected_points);
}

TEST(kd_tree_naive, kd_tree_naive_timeing)
{
  int i, j, n = 65536;
  float *points;
  points = (float*) malloc(n * 3 * sizeof(float));
  srand(time(NULL));
  for ( i = 0; i < n; ++i)
  {
    for ( j = 0; j < 3; ++j)
    {
      points[h_index(i, j, n)] = (float) rand() /100000000;
    }
  }

  cudaEvent_t start, stop;
  unsigned int bytes = n*3 * (sizeof(float));
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));
  float elapsed_time=0;

  checkCudaErrors(cudaEventRecord(start, 0));

  build_kd_tree(points, n);

  checkCudaErrors(cudaEventRecord(stop, 0));
  cudaEventSynchronize(start);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_time, start, stop);
  elapsed_time = elapsed_time ;
  double throughput = 1.0e-9 * ((double)bytes)/(elapsed_time* 1e-3);
  printf("build_kd_tree_naive, Throughput = %.4f GB/s, Time = %.5f ms, Size = %u Elements, NumDevsUsed = %d\n",
    throughput, elapsed_time, n, 1);

  free(points);
}


// float *ref, *dist;
// float *query;
// int *ind;
// unsigned int    ref_nb = 131072;
// unsigned int    query_nb = 1;
// unsigned int    dim=3;
// unsigned int    k          = 100;
// unsigned int    iterations = 1;
// unsigned int    i;

// ref    = (float *) malloc(ref_nb   * dim * sizeof(float));
// query  = (float *) malloc(query_nb * dim * sizeof(float));
// dist  = (float *) malloc(k * sizeof(float));
// ind  = (int *) malloc(k * sizeof(float));

// for (unsigned int count = 0; count < ref_nb*dim; count++)
// {
//   ref[count] = (float)ref_nb*dim-count;
// }
// for (unsigned int count = 0; count < query_nb*dim; count++)
// {
//   query[count] = 0;
// }

// for (i=0; i<iterations; i++){
//   knn_brute_force_reduce(ref, ref_nb, query, dim, k, dist, ind);
// }

// for (unsigned int i = 0; i < k; ++i)
// {
//   ASSERT_EQ(ind[i], ref_nb-1-i) << "Faild with i = "<<i << " and n = " << ref_nb;;
// }

// free(dist);
// free(ind);
// free(query);
// free(ref);
// cudaDeviceSynchronize();
// cudaDeviceReset();

#include <kd-tree-naive.cuh>
#include <knn_gpgpu.h>
#include <point.h>

#include <stdio.h>
#include <helper_cuda.h>
#include "gtest/gtest.h"


#define debug 0

int h_index(int i, int j, int n)
{
    return i + j * n;
}

void h_swap(Point *points, int a, int b, int n)
{
    Point t = points[a];
    points[a] = points[b], points[b] = t;
}

int midpoint(int lower, int upper)
{
    return (int) floor((upper - lower) / 2) + lower;
}

void print_tree(Point *tree, int level, int lower, int upper, int n)
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
    printf("(%3.1f, %3.1f, %3.1f)\n", tree[r].p[0], tree[r].p[1], tree[r].p[2]);

    print_tree(tree, 1 + level, lower, r, n);
    print_tree(tree, 1 + level, r + 1, upper, n);
  }
}

TEST(kd_tree_naive, kd_tree_naive_correctness){
  int i, j, n = 8;
  float temp;
  Point *points, *expected_points;
  points = (Point*) malloc(n  * sizeof(Point));
  expected_points = (Point*) malloc(n * sizeof(Point));
  srand(time(NULL));
  for ( i = 0; i < n; ++i)
  {
      temp = n-i-1;
      points[i] =(Point) {.p={temp,temp,temp}};
      expected_points[i] = (Point) {.p={i,i,i}};;
  }
  if (debug)
  {
    printf("kd tree expected:\n");
    print_tree(expected_points, 0, 0, n, n);
    printf("==================\n");

    printf("kd tree:\n");
    print_tree(points, 0, 0, n, n);
    printf("==================\n");

    for (int i = 0; i < n; ++i)
    {
      printf("%3.1f, ", points[i].p[0]);
    }
    printf("\n");
  }

  build_kd_tree(points, n);

  for ( i = 0; i < n; ++i)
  {
    for ( j = 0; j < 3; ++j)
    {
      ASSERT_EQ(points[i].p[j] ,expected_points[i].p[j]) << "Faild with i = " << i << " j = " <<j ;
    }
  }
  free(points);
  free(expected_points);
}

TEST(kd_tree_naive, kd_tree_naive_time){
  int i, n = 8388608/16;
  float temp;
  Point *points;
  points = (Point*) malloc(n  * sizeof(Point));
  srand(time(NULL));
  for ( i = 0; i < n; ++i)
  {
      temp = n-i-1;
      points[i] =(Point) {.p={temp,temp,temp}};
  }

  cudaEvent_t start, stop;
  unsigned int bytes = n * (sizeof(Point));
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



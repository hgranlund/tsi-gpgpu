#include "kd-tree-naive.cuh"
#include "knn_gpgpu.h"
#include "point.h"

#include "stdio.h"
#include "helper_cuda.h"
#include "gtest/gtest.h"






#define debug 0



__host__  void h_printPointsArray__(Point *l, int n, char *s, int l_debug=0)
{
  if (debug || l_debug)
  {
    printf("%10s: [ ", s);
      for (int i = 0; i < n; ++i)
      {
        printf("%3.1f, ", l[i].p[0]);
      }
      printf("]\n");
    }
  }


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
    return (int) floor((float)(upper - lower) / 2) + lower;
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

  int main(int argc, char const *argv[])
  {
    int i,n, nu, ni = 8388608,
    step = 250000;
    n=nu=ni;
    if(argc == 2) {
      nu = ni = atoi(argv[1]);
      printf("Running kd-tree-build with n = %d\n",nu);
    }
    else if(argc == 4) {
      nu = atoi(argv[1]);
      ni = atoi(argv[2]);
      step = atoi(argv[3]);
      printf("Running kd-tree-build from n = %d to n = %d with step = %d\n",nu,ni,step);
    }
    else{
      printf("Running kd-tree-build with n = %d\n",nu);
    }

    for (n = nu; n <=ni ; n+=step)
    {
      cudaDeviceReset();
      float temp;
      Point *points;
      points = (Point*) malloc(n  * sizeof(Point));
      srand(time(NULL));
      for ( i = 0; i < n; ++i)
      {
        temp = n-i-1;
        Point t;
        t.p[0]=temp;
        t.p[1]=temp;
        t.p[2]=temp;
        points[i]    = t;

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
    return 0;

  }

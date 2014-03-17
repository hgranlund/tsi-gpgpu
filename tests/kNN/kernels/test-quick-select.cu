// Includes
#include <quick-select.cuh>
#include <kd-tree-naive.cuh>
#include <knn_gpgpu.h>
#include <stdio.h>
#include <gtest/gtest.h>
#include <math.h>
#include <cuda.h>
#include <time.h>
#include <assert.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#define inf 0x7f800000
#define THREADS_PER_BLOCK 1024U
#define MAX_BLOCK_DIM_SIZE 65535U

#define debug 0

#define checkCudaErrors(val)           check ( (val), #val, __FILE__, __LINE__ )
#define FILE (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define debugf(fmt, ...) if(debug)printf("%s:%d: " fmt, FILE, __LINE__, __VA_ARGS__);



__host__  void h_printPointsArray_(Point *l, int n, char *s, int l_debug=0)
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

  TEST(kernels, quick_selection){
    Point *h_points, *d_points;
    int numBlocks, numThreads;
    float temp;
    unsigned int i,n, p;
    for (n = 8; n <=9; n<<=1)
    {
      p=16;
      numBlocks = p;
      n=n*p;
      h_points = (Point*) malloc(n*sizeof(Point));
      srand ( (unsigned int)time(NULL) );
      for (i=0 ; i<n; i++)
      {
        temp =  (float) rand()/100000000;
        Point t;
        t.p[0]=temp;
        t.p[1]=temp;
        t.p[2]=temp;
        h_points[i]    = t;
      }
      getThreadAndBlockCountForQuickSelect(n, p, numBlocks, numThreads);

      checkCudaErrors(
        cudaMalloc((void **)&d_points, n*sizeof(Point)));
      checkCudaErrors(
        cudaMemcpy(d_points, h_points, n*sizeof(Point), cudaMemcpyHostToDevice));

      h_printPointsArray_(h_points,n, "h_points",0);
      cuQuickSelectShared<8><<<2,2>>>(d_points, n/p, p, 0);

      checkCudaErrors(
        cudaMemcpy(h_points, d_points, n*sizeof(Point), cudaMemcpyDeviceToHost));

      h_printPointsArray_(h_points,n, "h_points",0);

      int step = n/p;
      Point *t_points;
      for (int i = 0; i < p; ++i)
      {
        t_points = h_points+i*step;
        for (int i = 0; i < step/2; ++i)
        {
          ASSERT_LE(t_points[i].p[0], t_points[step/2].p[0]) << "Faild with n = " << n;
        }
        for (int i = step/2; i < step; ++i)
        {
          ASSERT_GE(t_points[i].p[0], t_points[step/2].p[0]) << "Faild with n = " << n;
        }
      }

      checkCudaErrors(
        cudaFree(d_points));
      free(h_points);
      cudaDeviceSynchronize();
      cudaDeviceReset();
    }
  }
  TEST(kernels, quick_selection_time){
    Point *h_points, *d_points;
    int numBlocks, numThreads;
    float temp;
    unsigned int i,n, p;
    for (n = 8; n <=9; n<<=1)
    {
      p=1048576;
      numBlocks = p;
      n=n*p;
      h_points = (Point*) malloc(n*sizeof(Point));
      srand ( (unsigned int)time(NULL) );
      for (i=0 ; i<n; i++)
      {
        temp =  (float) rand()/100000000;
        Point t;
        t.p[0]=temp;
        t.p[1]=temp;
        t.p[2]=temp;
        h_points[i]    = t;
      }
      getThreadAndBlockCountForQuickSelect(n, p, numBlocks, numThreads);

      checkCudaErrors(
        cudaMalloc((void **)&d_points, n*sizeof(Point)));
      checkCudaErrors(
        cudaMemcpy(d_points, h_points, n*sizeof(Point), cudaMemcpyHostToDevice));


      cudaEvent_t start, stop;
      unsigned int bytes = n * (sizeof(Point));
      checkCudaErrors(cudaEventCreate(&start));
      checkCudaErrors(cudaEventCreate(&stop));
      float elapsed_time=0;
      checkCudaErrors(cudaEventRecord(start, 0));

      cuQuickSelectGlobal<<<numBlocks,numThreads>>>(d_points, n/p, p, 0);

      checkCudaErrors(cudaEventRecord(stop, 0));
      cudaEventSynchronize(start);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&elapsed_time, start, stop);
      elapsed_time = elapsed_time ;
      double throughput = 1.0e-9 * ((double)bytes)/(elapsed_time* 1e-3);
      printf("quick_selection, Throughput = %.4f GB/s, Time = %.5f ms, Size = %d, p = %d, NumDevsUsed = %d\n",
       throughput, elapsed_time, n, p, 1);

      checkCudaErrors(
        cudaMemcpy(h_points, d_points, n*sizeof(Point), cudaMemcpyDeviceToHost));

      int step = n/p;
      Point *t_points;
      for (int i = 0; i < p; ++i)
      {
        t_points = h_points+i*step;
        for (int i = 0; i < step/2; ++i)
        {
          ASSERT_LE(t_points[i].p[0], t_points[step/2].p[0]) << "Faild with n = " << n;
        }
        for (int i = step/2; i < step; ++i)
        {
          ASSERT_GE(t_points[i].p[0], t_points[step/2].p[0]) << "Faild with n = " << n;
        }
      }

      checkCudaErrors(
        cudaFree(d_points));
      free(h_points);
      cudaDeviceSynchronize();
      cudaDeviceReset();
    }
  }

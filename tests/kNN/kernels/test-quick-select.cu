// Includes
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

#define debug 1

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
    int *h_blockOffsets, *d_blockOffsets;
    float temp;
    unsigned int i,n, p;
    for (n = 8; n <=8; n<<=1)
    {
      p=5;
      int numBlocks = p;
      n=n*p;
      h_points = (Point*) malloc(n*sizeof(Point));
      srand ( (unsigned int)time(NULL) );
      for (i=0 ; i<n; i++)
      {
        temp =  (float) rand()/100000000;
        h_points[i]    = (Point) {temp, temp, temp};
      }

      h_blockOffsets = (int*) malloc((numBlocks+1)*sizeof(int));
      h_blockOffsets[numBlocks]=numBlocks;
      h_blockOffsets[0]=0;
      int total_lists = n/(n/p);
      for (int i = 1; i < numBlocks; ++i)
      {
        h_blockOffsets[i]=total_lists/numBlocks * i;
      }

      checkCudaErrors(
        cudaMalloc((void **)&d_points, n*sizeof(Point)));
      checkCudaErrors(
        cudaMemcpy(d_points, h_points, n*sizeof(Point), cudaMemcpyHostToDevice));

      checkCudaErrors(
        cudaMalloc((void **)&d_blockOffsets, (numBlocks+1)*sizeof(int)));
      checkCudaErrors(
        cudaMemcpy(d_blockOffsets, h_blockOffsets, (numBlocks+1)*sizeof(int), cudaMemcpyHostToDevice));


      cuQuickSelect<<<numBlocks,512>>>(d_points, n, p, d_blockOffsets, 0);

      checkCudaErrors(
        cudaMemcpy(h_points, d_points, n*sizeof(Point), cudaMemcpyDeviceToHost));

      h_printPointsArray_(h_points,n, "h_points");

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
      checkCudaErrors(
        cudaFree(d_blockOffsets));
      free(h_points);
      free(h_blockOffsets);
      cudaDeviceSynchronize();
      cudaDeviceReset();
    }
  }
  TEST(kernels, quick_selection_time){
    Point *h_points, *d_points;
    int *h_blockOffsets, *d_blockOffsets;
    float temp;
    unsigned int i,n, p;
    for (n = 8; n <=8; n<<=1)
    {
      p=5;
      int numBlocks = p;
      n=n*p;
      h_points = (Point*) malloc(n*sizeof(Point));
      srand ( (unsigned int)time(NULL) );
      for (i=0 ; i<n; i++)
      {
        temp =  (float) rand()/100000000;
        h_points[i]    = (Point) {temp, temp, temp};
      }

      h_blockOffsets = (int*) malloc((numBlocks+1)*sizeof(int));
      h_blockOffsets[numBlocks]=numBlocks;
      h_blockOffsets[0]=0;
      int total_lists = n/(n/p);
      for (int i = 1; i < numBlocks; ++i)
      {
        h_blockOffsets[i]=total_lists/numBlocks * i;
      }

      checkCudaErrors(
        cudaMalloc((void **)&d_points, n*sizeof(Point)));
      checkCudaErrors(
        cudaMemcpy(d_points, h_points, n*sizeof(Point), cudaMemcpyHostToDevice));

      checkCudaErrors(
        cudaMalloc((void **)&d_blockOffsets, (numBlocks+1)*sizeof(int)));
      checkCudaErrors(
        cudaMemcpy(d_blockOffsets, h_blockOffsets, (numBlocks+1)*sizeof(int), cudaMemcpyHostToDevice));

      cudaEvent_t start, stop;
      unsigned int bytes = n * (sizeof(Point));
      checkCudaErrors(cudaEventCreate(&start));
      checkCudaErrors(cudaEventCreate(&stop));
      float elapsed_time=0;

      checkCudaErrors(cudaEventRecord(start, 0));


      cuQuickSelect<<<numBlocks,512>>>(d_points, n, p, d_blockOffsets, 0);

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


      checkCudaErrors(
        cudaFree(d_points));
      checkCudaErrors(
        cudaFree(d_blockOffsets));
      free(h_points);
      free(h_blockOffsets);
      cudaDeviceSynchronize();
      cudaDeviceReset();
    }
  }

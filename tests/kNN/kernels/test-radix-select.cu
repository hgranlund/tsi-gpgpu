// Includes
#include <radix-select.cuh>
#include <knn_gpgpu.h>
#include <stdio.h>
#include <gtest/gtest.h>

#include <math.h>
#include <cuda.h>
#include <time.h>
#include <assert.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#define checkCudaErrors(val)           check ( (val), #val, __FILE__, __LINE__ )
#define inf 0x7f800000

#define debug 0
#define FILE (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define debugf(fmt, ...) if(debug)printf("%s:%d: " fmt, FILE, __LINE__, __VA_ARGS__);


void printPoints(Point* l, int n){
  int i;
  if (debug)
  {
    printf("[(%3.1f, %3.1f, %3.1f)", l[0].p[0], l[0].p[1], l[0].p[2]);
      for (i = 1; i < n; ++i)
      {
        printf(", (%3.1f, %3.1f, %3.1f)", l[i].p[0], l[i].p[1], l[i].p[2]);
      }
      printf("]\n");
    }
  }


  TEST(kernels, radix_selection){
    Point *h_points;
    float temp;
    unsigned int i,n;
    for (n = 4; n <=200; n+=10)
    {
      h_points = (Point*) malloc(n*sizeof(Point));
      srand ( (unsigned int)time(NULL) );
      for (i=0 ; i<n; i++)
      {
        temp =  (float) rand()/100000000;
        h_points[i]    = (Point) {temp, temp, temp};
      }

      printPoints(h_points,n);

      Point *d_points, *d_temp, *d_result, h_result;
      int *partition;
      checkCudaErrors(
        cudaMalloc((void **)&d_result, sizeof(Point)));
      checkCudaErrors(
        cudaMalloc((void **)&d_points, n*sizeof(Point)));
      checkCudaErrors(
        cudaMalloc((void **)&d_temp, n*sizeof(Point)));
      checkCudaErrors(
        cudaMalloc((void **)&partition, n*sizeof(int)));
      checkCudaErrors(
        cudaMemcpy(d_points, h_points, n*sizeof(Point), cudaMemcpyHostToDevice));


      Point cpu_result = cpu_radixselect(h_points, 0, n-1, n/2, 0);

      cuRadixSelect<<<1,64>>>(d_points, d_temp, n/2, n, partition, 0, d_result);
      checkCudaErrors(
       cudaMemcpy(&h_result, d_result, sizeof(Point), cudaMemcpyDeviceToHost));

      checkCudaErrors(
        cudaMemcpy(h_points, d_points, n*sizeof(Point), cudaMemcpyDeviceToHost));

      printPoints(h_points,n);
      debugf("result = (%3.1f, %3.1f, %3.1f)\n", h_result.p[0], h_result.p[1], h_result.p[2] );

        // printDistArray(h_points,n);
      ASSERT_EQ(cpu_result.p[0], h_result.p[0]) << "Faild with n = " << n;
      ASSERT_EQ(cpu_result.p[1], h_result.p[1]) << "Faild with n = " << n;
      ASSERT_EQ(cpu_result.p[2], h_result.p[2]) << "Faild with n = " << n;
      checkCudaErrors(
        cudaFree(d_points));
      checkCudaErrors(
        cudaFree(partition));
      checkCudaErrors(
        cudaFree(d_result));
      free(h_points);
      cudaDeviceSynchronize();
      cudaDeviceReset();
    }
  }

  TEST(kernels, radix_selection_time){
    Point *h_points;
    unsigned int i,n;
    n = 160000;

    h_points = (Point*) malloc(n*sizeof(Point));

    float temp;

    h_points = (Point*) malloc(n*sizeof(Point));
    srand ( (unsigned int)time(NULL) );
    for (i=0 ; i<n; i++)
    {
      temp =  (float) rand()/100000000;
      h_points[i]    = (Point) {temp, temp, temp};
    }

    printPoints(h_points,n);

    Point *d_points, *d_temp, *d_result, h_result;
    int *partition;
    checkCudaErrors(
      cudaMalloc((void **)&d_result, sizeof(Point)));
    checkCudaErrors(
      cudaMalloc((void **)&d_points, n*sizeof(Point)));
    checkCudaErrors(
      cudaMalloc((void **)&d_temp, n*sizeof(Point)));
    checkCudaErrors(
      cudaMalloc((void **)&partition, n*sizeof(int)));
    checkCudaErrors(
      cudaMemcpy(d_points, h_points, n*sizeof(Point), cudaMemcpyHostToDevice));



    cudaEvent_t start, stop;
    unsigned int bytes = n * (sizeof(float)) ;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    float elapsed_time=0;

    checkCudaErrors(cudaEventRecord(start, 0));

    cuRadixSelect<<<1,1024>>>(d_points, d_temp, n/2, n, partition, 0, d_result);


    checkCudaErrors(cudaEventRecord(stop, 0));
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    elapsed_time = elapsed_time ;
    double throughput = 1.0e-9 * ((double)bytes)/(elapsed_time* 1e-3);
    printf("radix-select, Throughput = %.4f GB/s, Time = %.5f ms, Size = %u Elements, NumDevsUsed = %d\n",
     throughput, elapsed_time, n, 1);

    checkCudaErrors(
     cudaMemcpy(&h_result, d_result, sizeof(Point), cudaMemcpyDeviceToHost));

    checkCudaErrors(
      cudaMemcpy(h_points, d_points, n*sizeof(Point), cudaMemcpyDeviceToHost));

    printPoints(h_points,n);

    checkCudaErrors(
      cudaFree(d_points));
    checkCudaErrors(
      cudaFree(partition));
    checkCudaErrors(
      cudaFree(d_result));
    free(h_points);
    cudaDeviceSynchronize();
    cudaDeviceReset();
  }


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


void printPoints(float* l, int n){
  int i;
  if (debug)
  {
    printf("[%3.1f", l[0] );
      for (i = 1; i < n; ++i)
      {
        printf(", %3.1f", l[i] );
      }
      printf("]\n");
    }
  }


  TEST(kernels, radix_selection){
    float *h_points;
    unsigned int i,n;
    for (n = 100; n <=200; n+=10)
    {
      h_points = (float*) malloc(n*sizeof(float));
      srand ( (unsigned int)time(NULL) );
      for (i=0 ; i<n; i++)
      {
        h_points[i]    = n-1-i;
        h_points[i]    = i;
        h_points[i]    =(float) rand()/10000000;
      }

      printPoints(h_points,n);

      float *d_points, *d_temp, *d_result, h_result;
      int *d_ones, *d_zeros;
      h_result = 0;
      checkCudaErrors(
        cudaMalloc((void **)&d_result, sizeof(float)));
      checkCudaErrors(
        cudaMalloc((void **)&d_points, n*sizeof(float)));
      checkCudaErrors(
        cudaMalloc((void **)&d_temp, n*sizeof(float)));
      checkCudaErrors(
        cudaMalloc((void **)&d_ones, n*sizeof(int)));
      checkCudaErrors(
        cudaMalloc((void **)&d_zeros, n*sizeof(int)));

      checkCudaErrors(
        cudaMemcpy(d_points, h_points, n*sizeof(float), cudaMemcpyHostToDevice));


      float cpu_result = cpu_radixselect(h_points, 0, n-1, n/2, 0);

      cuRadixSelect<<<1,64>>>(d_points, d_temp, n/2, n, d_ones, d_zeros, d_result);
      checkCudaErrors(
       cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost));

      checkCudaErrors(
        cudaMemcpy(h_points, d_points, n*sizeof(float), cudaMemcpyDeviceToHost));

      printPoints(h_points,n);
      debugf("result = %3.1f\n", h_result);

        // printDistArray(h_points,n);
      ASSERT_EQ(cpu_result, h_result) << "Faild with n = " << n;
      checkCudaErrors(
        cudaFree(d_points));
      checkCudaErrors(
        cudaFree(d_ones));
      checkCudaErrors(
        cudaFree(d_zeros));
      checkCudaErrors(
        cudaFree(d_result));
      free(h_points);
      cudaDeviceSynchronize();
      cudaDeviceReset();
    }
  }
  TEST(kernels, radix_selection_time){
    float *h_points;
    unsigned int i,n;
    n = 160000;

    h_points = (float*) malloc(n*sizeof(float));

    srand ( (unsigned int)time(NULL) );
    for (i=0 ; i<n; i++)
    {
      h_points[i]    = n-1-i;
      h_points[i]    =(float) rand()/10000000;
      h_points[i]    = i;
    }

    printPoints(h_points,n);

    float *d_points, *d_temp, *d_result, h_result;
    int *d_ones, *d_zeros;
    h_result = 0;
    checkCudaErrors(
      cudaMalloc((void **)&d_result, sizeof(float)));
    checkCudaErrors(
      cudaMalloc((void **)&d_points, n*sizeof(float)));
    checkCudaErrors(
      cudaMalloc((void **)&d_temp, n*sizeof(float)));
    checkCudaErrors(
      cudaMalloc((void **)&d_ones, n*sizeof(int)));
    checkCudaErrors(
      cudaMalloc((void **)&d_zeros, n*sizeof(int)));
    checkCudaErrors(
      cudaMemcpy(d_points, h_points, n*sizeof(float), cudaMemcpyHostToDevice));



    cudaEvent_t start, stop;
    unsigned int bytes = n * (sizeof(float)+sizeof(int));
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    float elapsed_time=0;

    checkCudaErrors(cudaEventRecord(start, 0));

    cuRadixSelect<<<1,1024>>>(d_points, d_temp, n/2, n, d_ones, d_zeros, d_result);


    checkCudaErrors(cudaEventRecord(stop, 0));
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    elapsed_time = elapsed_time ;
    double throughput = 1.0e-9 * ((double)bytes)/(elapsed_time* 1e-3);
    printf("radix-select, Throughput = %.4f GB/s, Time = %.5f ms, Size = %u Elements, NumDevsUsed = %d\n",
     throughput, elapsed_time, n, 1);

    checkCudaErrors(
     cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost));

    checkCudaErrors(
      cudaMemcpy(h_points, d_points, n*sizeof(float), cudaMemcpyDeviceToHost));

    printPoints(h_points,n);

        // printDistArray(h_points,n);
    checkCudaErrors(
      cudaFree(d_points));
    checkCudaErrors(
      cudaFree(d_ones));
    checkCudaErrors(
      cudaFree(d_zeros));
    checkCudaErrors(
      cudaFree(d_result));
    free(h_points);
    cudaDeviceSynchronize();
    cudaDeviceReset();
  }


  // TEST(kernels, radix_selection_time){
  //   cudaDeviceReset();

  //   float *h_points;
  //   unsigned int i,n;
  //   n=2048;
  //   h_points = (float*) malloc(n*sizeof(float));

  //   srand ( (unsigned int)time(NULL) );
  //   for (i=0 ; i<n; i++)
  //   {
  //     h_points[i]    =(float) rand()/10000000;
  //   }

  //   float *d_points, *d_result, h_result;
  //   h_result = 0;
  //   checkCudaErrors(
  //     cudaMalloc((void **)&d_result, sizeof(float)));
  //   checkCudaErrors(
  //     cudaMalloc((void **)&d_points, n*sizeof(float)));

  //   checkCudaErrors(
  //     cudaMemcpy(d_points, h_points, n*sizeof(float), cudaMemcpyHostToDevice));

  //   cudaEvent_t start, stop;
  //   unsigned int bytes = n * (sizeof(float)+sizeof(int));
  //   checkCudaErrors(cudaEventCreate(&start));
  //   checkCudaErrors(cudaEventCreate(&stop));
  //   float elapsed_time=0;

  //   checkCudaErrors(cudaEventRecord(start, 0));


  //   // cuRadixSelect<<<1,1024>>>(d_points, n/2, n, d_result);

  //   checkCudaErrors(cudaEventRecord(stop, 0));
  //   cudaEventSynchronize(start);
  //   cudaEventSynchronize(stop);
  //   cudaEventElapsedTime(&elapsed_time, start, stop);
  //   elapsed_time = elapsed_time ;
  //   double throughput = 1.0e-9 * ((double)bytes)/(elapsed_time* 1e-3);
  //   printf("radix-select, Throughput = %.4f GB/s, Time = %.5f ms, Size = %u Elements, NumDevsUsed = %d\n",
  //    throughput, elapsed_time, n, 1);



  //   checkCudaErrors(
  //    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost));

  //   checkCudaErrors(
  //     cudaMemcpy(h_points, d_points, n*sizeof(float), cudaMemcpyDeviceToHost));

  //   printPoints(h_points,n);
  //   cudaFree(d_points);
  //   cudaFree(d_result);
  //   free(h_points);
  //   cudaDeviceSynchronize();
  //   cudaDeviceReset();
  // }


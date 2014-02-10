// Includes
#include <reduction.cuh>
#include <knn_gpgpu.h>
#include <stdio.h>
#include <gtest/gtest.h>

#include <math.h>
#include <cuda.h>
#include <time.h>
#include <assert.h>




#define inf 0x7f800000
#define debug 1
float cpu_min(float* in, int num_els)
{
  float min = inf;

  for(int i = 0; i < num_els; i++)
    min = in[i] < min ? in[i] : min;

  return min;
}

void printFloatArray(float* l, int n){
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

  void printIntArray(int* l, int n){
    int i;
    if (debug)
    {
      printf("[%4d", l[0] );
        for (i = 1; i < n; ++i)
        {
          printf(", %4d", l[i] );
        }
        printf("]\n");
      }
    }


    TEST(kernels, min_reduce){

      float *h_list;
      int *h_ind;
      int i,n;
      for (n = 11; n <=11; n +=2)
      {

        h_list = (float*) malloc(n*sizeof(float));
        h_ind = (int*) malloc(n*sizeof(int));
        srand(time(NULL));
        for (i=0 ; i<n; i++)
        {
          h_list[i]    = n-i-1;
          h_ind[i]=i;
        }
        // printf("########\n");
        // printFloatArray(list,n);
        // printIntArray(ind_1,n);

        float *d_list;
        int *d_ind;

        cudaMalloc( (void **) &d_list, n* sizeof(float));
        cudaMalloc( (void **) &d_ind, n* sizeof(int));

        cudaMemcpy(d_list,h_list, n*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_ind,h_ind, n*sizeof(int), cudaMemcpyHostToDevice);

        knn_min_reduce(d_list, d_ind, n);

        cudaMemcpy(h_list,d_list, n*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_ind,d_ind, n*sizeof(int), cudaMemcpyDeviceToHost);

        ASSERT_LE(h_list[0], 0)  << "Faild with n = " << n;
        ASSERT_LE(h_ind[0], n-1)  << "Faild with n = " << n;

        cudaFree(d_list);
        cudaFree(d_ind);
        free(h_list);
        free(h_ind);
      }
    }


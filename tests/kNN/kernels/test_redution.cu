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

void printFloatArray(float* l, int n)
{
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

      Distance *h_dist;
      int i,n;
      for (n = 11; n <=11; n +=2)
      {

        h_dist = (Distance*) malloc(n*sizeof(Distance));

        srand(time(NULL));
        for (i=0 ; i<n; i++)
        {
          h_dist[i].value    = n-i-1;
          h_dist[i].value=i;
        }
        // printf("########\n");
        // printFloatArray(list,n);
        // printIntArray(ind_1,n);

        Distance *d_dist;

        cudaMalloc( (void **) &d_dist, n* sizeof(Distance));

        cudaMemcpy(d_dist,h_dist, n*sizeof(Distance), cudaMemcpyHostToDevice);

        knn_min_reduce(d_dist, n);

        cudaMemcpy(h_dist,d_dist, n*sizeof(Distance), cudaMemcpyDeviceToHost);

        ASSERT_LE(h_dist[0].value, 0)  << "Faild with n = " << n;
        ASSERT_LE(h_dist[0].index, n-1)  << "Faild with n = " << n;

        cudaFree(d_dist);
        free(h_dist);
      }
    }


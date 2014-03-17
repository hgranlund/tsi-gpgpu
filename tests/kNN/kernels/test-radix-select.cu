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
#define THREADS_PER_BLOCK 1024U
#define MAX_BLOCK_DIM_SIZE 65535U
#define debug 0
#define FILE (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define debugf(fmt, ...) if(debug)printf("%s:%d: " fmt, FILE, __LINE__, __VA_ARGS__);


int cpu_partition1(Point *data, int l, int u, int bit)
{
  unsigned int radix=(1 << 31-bit);
  Point *temp = (Point *)malloc(((u-l)+1)*sizeof(Point));
  int pos = 0;
  for (int i = l; i<=u; i++)
  {
    if(((*(int*)&(data[i].p[0]))&radix))
    {
      temp[pos++] = data[i];
    }
  }
  int result = u-pos;
  for (int i = l; i<=u; i++)
  {
    if(!((*(int*)&(data[i]))&radix))
    {
      temp[pos++] = data[i];
    }
  }
  pos = 0;
  for (int i = u; i>=l; i--)
  {
    data[i] = temp[pos++];
  }

  free(temp);
  return result;
}

Point cpu_radixselect1(Point *data, int l, int u, int m, int bit){

  Point t;
  t.p[0] = 0;
  t.p[1] = 0;
  t.p[2] = 0;
  if (l == u) return(data[l]);
  if (bit > 32) {printf("cpu_radixselect1 fail!\n"); return t;}
  int s = cpu_partition1(data, l, u, bit);
  if (s>=m) return cpu_radixselect1(data, l, s, m, bit+1);
  return cpu_radixselect1(data, s+1, u, m, bit+1);
}






void printPoints1(Point* l, int n)
{
  int i;
  if (debug)
  {
// printf("[(%3.1f, %3.1f, %3.1f)", l[0].p[0], l[0].p[1], l[0].p[2]);
    printf("[%3.1f, ", l[0].p[0]);
      for (i = 1; i < n; ++i)
      {
        printf(", %3.1f, ", l[i].p[0]);
// printf(", (%3.1f, %3.1f, %3.1f)", l[i].p[0], l[i].p[1], l[i].p[2]);
      }
      printf("]\n");
    }
  }

  __device__ __host__
  unsigned int nextPowerOf22(unsigned int x)
  {
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
  }

  __device__ __host__
  bool isPowTwo2(unsigned int x)
  {
    return ((x&(x-1))==0);
  }

  __device__ __host__
  unsigned int prevPowerOf22(unsigned int n)
  {
    if (isPowTwo2(n))
    {
      return n;
    }
    n = nextPowerOf22(n);
    return n >>=1;

  }

  TEST(kernels, radix_selection){
    Point *h_points;
    float temp;
    int i,n;
    for (n = 4; n <=1000; n<<=1)
    {
      h_points = (Point*) malloc(n*sizeof(Point));
      srand ( (unsigned int)time(NULL) );
      for (i=0 ; i<n; i++)
      {
        temp =  (float) n-1-i;
        temp =  (float) rand()/100000000;
        Point t;
        t.p[0]=temp;
        t.p[1]=temp;
        t.p[2]=temp;
        h_points[i]    = t;
      }

      printPoints1(h_points,n);

      Point *d_points, *d_temp;
      int *partition;
      checkCudaErrors(
        cudaMalloc((void **)&d_points, n*sizeof(Point)));
      checkCudaErrors(
        cudaMalloc((void **)&d_temp, n*sizeof(Point)));
      checkCudaErrors(
        cudaMalloc((void **)&partition, n*sizeof(int)));
      checkCudaErrors(
        cudaMemcpy(d_points, h_points, n*sizeof(Point), cudaMemcpyHostToDevice));


      Point cpu_result = cpu_radixselect1(h_points, 0, n-1, n/2, 0);

      radixSelectAndPartition(d_points, d_temp, partition, n/2, n, 0);

      checkCudaErrors(
        cudaMemcpy(h_points, d_points, n*sizeof(Point), cudaMemcpyDeviceToHost));

      printPoints1(h_points,n);

      debugf("result = (%3.1f, %3.1f, %3.1f)\n", h_points[n/2].p[0], h_points[n/2].p[1], h_points[n/2].p[2] );
      ASSERT_EQ(cpu_result.p[0], h_points[n/2].p[0]) << "Faild with n = " << n;
      ASSERT_EQ(cpu_result.p[1], h_points[n/2].p[1]) << "Faild with n = " << n;
      ASSERT_EQ(cpu_result.p[2], h_points[n/2].p[2]) << "Faild with n = " << n;

      for (int i = 0; i < n/2; ++i)
      {
        ASSERT_LE(h_points[i].p[0], h_points[n/2].p[0]) << "Faild with n = " << n;
      }
      for (int i = n/2; i < n; ++i)
      {
        ASSERT_GE(h_points[i].p[0], h_points[n/2].p[0]) << "Faild with n = " << n;
      }
      checkCudaErrors(
        cudaFree(d_points));
      checkCudaErrors(
        cudaFree(partition));
      cudaDeviceSynchronize();
      cudaDeviceReset();
    }
  }



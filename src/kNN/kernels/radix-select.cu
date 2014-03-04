#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"
#include <radix-select.cuh>

#define checkCudaErrors(val)           check ( (val), #val, __FILE__, __LINE__ )
#define inf 0x7f800000

#define debug 1
#define FILE (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define debugf(fmt, ...) if(debug)printf("%s:%d: " fmt, FILE, __LINE__, __VA_ARGS__);



__device__ void cuSwap(float *data, int a, int b)
{
  float temp = data[a];
  data[a]=data[b];
  data[b]=temp;
}

__device__ void printArray(float *l, int n)
{
  if (debug)
  {
#if __CUDA_ARCH__>=200
    if (threadIdx.x == 0)
    {
      printf("[ ");
       for (int i = 0; i < n; ++i)
       {
        printf("%3.1f, ", l[i]);
      }
      printf("]\n");
    }
    __syncthreads();
  #endif
  }
}

__device__ unsigned int sum_reduce(int *list, int n)
{
  int half = n/2;
  int tid = threadIdx.x;
  while(tid<half && half > 0)
  {
    list[tid] += list[tid+half];
    half = half/2;
  }
  return list[0];
}


__device__ unsigned int cuPartition(float *data, unsigned int n, float *ones, float *zeros, int *zero_count, int *one_count, unsigned int bit)
{
  unsigned int cut = 0,
  tid = threadIdx.x,
  radix = (1 << 31-bit);
  zero_count[threadIdx.x] = 0;
  one_count[threadIdx.x] = 0;
  while(tid < n)
  {
    if (((*(int*)&(data[tid]))&radix))
    {
      ones[tid]=data[tid];
      one_count[threadIdx.x] += 1;
    }
    tid+=blockDim.x;
  }
  tid = threadIdx.x;
  __syncthreads();
  while(tid < n)
  {
    if (!((*(int*)&(data[tid]))&radix))
    {
      zeros[tid]=data[tid];
      zero_count[threadIdx.x]+=1;
    }
    tid+=blockDim.x;
  }
  cut = sum_reduce(zero_count, blockDim.x);
  printArray(zeros, cut);
  printArray(ones, n-cut);
  tid = threadIdx.x;
  __syncthreads();
  while(tid<cut)
  {
    data[tid]=zeros[tid];
    tid+=blockDim.x;
  }
  tid = threadIdx.x;
  __syncthreads();
  while(tid<n-cut)
  {
    data[n-tid-1] = ones[tid];
    tid+=blockDim.x;
  }

  return cut;
}


__global__ void cuRadixSelect(float *data, unsigned int m, unsigned int n, float *ones, float *zeros, float *result)
{
  __shared__ int one_count[1024];
  __shared__ int zeros_count[1024];


  unsigned int l=0,
  u = n,
  cut=0,
  bit = 0,
  tid = threadIdx.x;
  if (n<2)
  {
    if ((n == 1) && !(tid))
    {
      *result = data[0];
    }
    return;
  }
  do {


    cut = cuPartition(data+l, u-l, ones, zeros, one_count, zeros_count, bit++);
    __syncthreads();

  #if __CUDA_ARCH__>=200
    if (tid==0)
    {
      printf("\n l = %d, u= %d cut = %d, bit =%d \n",l,u, cut, bit);
    }
    #endif
    if ((u-cut) >= m)
    {
      u -=cut;
    }
    else
    {
      l +=(u-cut);
    }

    printArray(data+l,u-l);
  }while ((u-l>1) && (bit<32));

  if (tid == 0)
  {
    *result = data[m];
  }
  free(ones);
  free(zeros);
  // if (next[threadIdx.x]) *result = loc_data[threadIdx.x];
}

float partition(float *data, int l, int u, int bit)
{
  unsigned int radix=(1 << 31-bit);
  float *temp = (float *)malloc(((u-l)+1)*sizeof(float));
  int pos = 0;
// printf("l = %d, u = %d, bit = %d\n", l,u,bit);
  for (int i = l; i<=u; i++)
  {
    if(((*(int*)&(data[i]))&radix))
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
// printf("temp : %2d:  %3.1f\n", i, data[i]);
  }

  free(temp);
  return result;
}

float cpu_radixselect(float *data, int l, int u, int m, int bit){

  if (l == u) return(data[l]);
  if (bit > 32) {printf("cpu_radixselect fail!\n"); return 0;}
  int s = partition(data, l, u, bit);
  if (s>=m) return cpu_radixselect(data, l, s, m, bit+1);
  return cpu_radixselect(data, s+1, u, m, bit+1);
}





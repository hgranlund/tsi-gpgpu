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

__device__ void printArray(float *l, int n, char *s)
{
  if (debug)
  {
        #if __CUDA_ARCH__>=200
    if (threadIdx.x == 0)
    {
      printf("%10s: [ ", s);
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

  __device__ void printArrayInt(int *l, int n, char *s)
  {
    if (debug)
    {
#if __CUDA_ARCH__>=200
      if (threadIdx.x == 0)
      {
        printf("%10s: [ ", s);
          for (int i = 0; i < n; ++i)
          {
            printf("%3d, ", l[i]);
          }
          printf("]\n");
        }
        __syncthreads();
#endif
      }
    }

    __device__ unsigned int cuSumReduce(int *list, int n)
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

//TODO must be imporved
    __device__  void cuAccumulateIndex(int *list, int n)
    {
      int tid = threadIdx.x;
      if (tid == 0)
      {
        int sum=0;
        list[n]=list[n-1];
        int temp=0;
        for (int i = 0; i < n; ++i)
        {
          temp = list[i];
          list[i] = sum;
          sum += temp;
        }
        list[n]+=list[n-1];
      }
      __syncthreads();
    }


    __device__ unsigned int cuPartition(float *data, float *data_copy, unsigned int n, int *ones, int *zeros, int *zero_count, int *one_count, unsigned int bit)
    {
      unsigned int cut = 0,
      tid = threadIdx.x,
      i,
      radix = (1 << 31-bit);
      zero_count[threadIdx.x] = 0;
      one_count[threadIdx.x] = 0;
      while(tid < n)
      {
        ones[tid] = 0;
        zeros[tid] = 0;
        data_copy[tid]=data[tid];
        if (((*(int*)&(data[tid]))&radix))
        {
          one_count[threadIdx.x] += 1;
          ones[tid]=one_count[threadIdx.x];
        }else{
          zero_count[threadIdx.x]+=1;
          zeros[tid]=zero_count[threadIdx.x];
        }
        tid+=blockDim.x;
      }
      __syncthreads();
      int last_zero_count = zero_count[blockDim.x-1];
      cuAccumulateIndex(zero_count, blockDim.x);
      cuAccumulateIndex(one_count, blockDim.x);

      tid = threadIdx.x;
      __syncthreads();
      i = zero_count[threadIdx.x];

      while(tid<n && i<zero_count[threadIdx.x+1])
      {
        if (zeros[tid])
        {
          data[i]=data_copy[tid];
          i++;
        }
        tid+=blockDim.x;
      }
      tid = threadIdx.x;
      i = one_count[threadIdx.x];
      while(tid<n && one_count[threadIdx.x+1]){
        if (ones[tid])
        {
          data[n-i-1]=data_copy[tid];
          i++;
        }
        tid+=blockDim.x;
      }
      cut = zero_count[blockDim.x-1]+last_zero_count;
      return cut;
    }

//TODO do not need ones and zeroes, only one partitian or store the partition data on dava/datacopy
    __global__ void cuRadixSelect(float *data, float *data_copy, unsigned int m, unsigned int n, int *ones, int *zeros, float *result)
    {
      __shared__ int one_count[2048];
      __shared__ int zeros_count[2048];

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

        cut = cuPartition(data+l, data_copy, u-l, ones, zeros, one_count, zeros_count, bit++);
        __syncthreads();
        if ((l+cut) <= m)
        {
          l +=(cut);
        }
        else
        {
          u -=(u-cut-l);
        }
      }while ((u > l) && (bit<32));
      if (tid == 0)
      {
        *result = data[m];
      }
    }

    float partition(float *data, int l, int u, int bit)
    {
      unsigned int radix=(1 << 31-bit);
      float *temp = (float *)malloc(((u-l)+1)*sizeof(float));
      int pos = 0;
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





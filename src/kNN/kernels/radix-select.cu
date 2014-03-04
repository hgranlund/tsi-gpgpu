#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"
#include <cuda_runtime_api.h>

__device__ int gpu_partition(float *data, float *partition, float *ones, float* zeroes, int bit, int idx, float* warp_ones)
{
  int one = 0;
  int valid = 0;
  int my_one, my_zero;
  if (partition[idx])
  {
    valid = 1;
    if((*(int*)&(data[idx]))&(1 << (31-bit)))
    {
      one=1;
    }
  }
  __syncthreads();
  if (valid)
  {
    if (one)
    {
      my_one=1;
      my_zero=0;
    }
    else
    {
      my_one=0;
      my_zero=1;
    }
  }
  else
  {
    my_one=0;
    my_zero=0;
  }
  ones[idx]=my_one;
  zeroes[idx]=my_zero;
  unsigned int warp_one = __popc(__ballot(my_one));
  if (!(threadIdx.x & 31))
  {
    warp_ones[threadIdx.x>>5] = warp_one;
  }
  __syncthreads();
// reduce
  for (int i = 16; i > 0; i>>=1)
  {
    if (threadIdx.x < i)
    {
      warp_ones[threadIdx.x] += warp_ones[threadIdx.x + i];
    }
    __syncthreads();
  }
  return warp_ones[0];
}

__global__ void gpu_radixkernel(float *data, unsigned int m, unsigned int n, float *result)
{
  __shared__ float loc_data[1024];
  __shared__ float loc_ones[1024];
  __shared__ float loc_zeroes[1024];
  __shared__ float loc_warp_ones[32];
  int l=0;
  int bit = 0;
  unsigned int u = n;
  if (n<2)
  {
    if ((n == 1) && !(threadIdx.x))
    {
      *result = data[0];
    }
    return;
  }
  loc_data[threadIdx.x] = data[threadIdx.x];
  loc_ones[threadIdx.x] = (threadIdx.x<n)?1:0;
  __syncthreads();
  float *next = loc_ones;
  do {
    int s = gpu_partition(loc_data, next, loc_ones, loc_zeroes, bit++, threadIdx.x, loc_warp_ones);
    if ((u-s) > m)
    {
      u = (u-s);
      next = loc_zeroes;
    }
    else
    {
      l = (u-s);
      next = loc_ones;
    }
  }while ((u != l) && (bit<32));
  if (next[threadIdx.x]) *result = loc_data[threadIdx.x];
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

float radixselect(float *data, int l, int u, int m, int bit){

  if (l == u) return(data[l]);
  if (bit > 32) {printf("radixselect fail!\n"); return 0;}
  int s = partition(data, l, u, bit);
  if (s>=m) return radixselect(data, l, s, m, bit+1);
  return radixselect(data, s+1, u, m, bit+1);
}

int main(int argc, char const *argv[])
{
  int n = 8;
  float data[8] = {32767, 88.2, 88.1, 44, 99, 101, 0.1, 7};
  float data1[n];

  for (int i = 0; i<n; i++)
  {
    for (int j=0; j<n; j++)
    {
      data1[j] = data[j];
    }
    printf("value[%d] = %3.1f\n", i, radixselect(data1, 0, n-1, i, 0));
  }

  float *d_data;
  cudaMalloc((void **)&d_data, 1024*sizeof(float));
  float *d_result;
  float h_result;
  cudaMalloc((void **)&d_result, sizeof(float));
  cudaMemcpy(d_data, data, 8*sizeof(float), cudaMemcpyHostToDevice);
  for (int i = 0; i < 8; i++){
    gpu_radixkernel<<<1,1024>>>(d_data, i, 8, d_result);
    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    printf("gpu result index %d = %3.1f\n", i, h_result);
  }

// unsigned int data2[1024];
// unsigned int data3[1024];

// for (int i = 0; i < 1024; i++) data2[i] = rand();
// cudaMemcpy(d_data, data2, 1024*sizeof(unsigned int), cudaMemcpyHostToDevice);
// for (int i = 0; i < 1024; i++){
//   for (int j = 0; j<1024; j++) data3[j] = data2[j];
//   // unsigned int cpuresult = radixselect(data3, 0, 1023, i, 0);
//   gpu_radixkernel<<<1,1024>>>(d_data, i, 1024, d_result);
//   cudaMemcpy(&h_result, d_result, sizeof(unsigned int), cudaMemcpyDeviceToHost);
//   if (h_result != cpuresult) {printf("mismatch at index %d, cpu: %d, gpu: %d\n", i, cpuresult, h_result); return 1;}
//   }
  printf("Finished\n");

  return 0;
}

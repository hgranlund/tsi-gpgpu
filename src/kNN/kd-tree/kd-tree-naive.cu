#include <radix-select.cuh>
#include <kd-tree-naive.cuh>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <point.h>
#include <helper_cuda.h>

#define checkCudaErrors(val)           check ( (val), #val, __FILE__, __LINE__ )
#define THREADS_PER_BLOCK 1024U
#define MAX_BLOCK_DIM_SIZE 65535U
// #define THREADS_PER_BLOCK 4U
// #define MAX_BLOCK_DIM_SIZE 8U

#include <string.h>
#define debug 0
#define FILE (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define debugf(fmt, ...) if(debug){printf("%s:%d: " fmt, FILE, __LINE__, __VA_ARGS__);}



__device__ void printArray(Point *l, int n, char *s)
{
  if (debug)
  {
        #if __CUDA_ARCH__>=200
    if (threadIdx.x == 0)
    {
      printf("%10s: [ ", s);
        for (int i = 0; i < n; ++i)
        {
          printf("(%3.1f, %3.1f, %3.1f), ", l[i].p[0], l[i].p[1], l[i].p[2]);
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
    if (threadIdx.x == 0)
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
}


__device__ unsigned int cuPartition(Point *data, Point *data_copy, unsigned int n, int *partition, int *zero_count, int *one_count, unsigned int bit, int dir)
{
    unsigned int
    tid = threadIdx.x,
    is_one,
    one,
    zero,
    radix = (1 << 31-bit);

    zero_count[threadIdx.x] = 0;
    one_count[threadIdx.x] = 0;

    while(tid < n)
    {
        data_copy[tid]=data[tid];
        is_one = partition[tid]= (bool)((*(int*)&(data[tid].p[dir]))&radix);
        one_count[threadIdx.x] += is_one;
        zero_count[threadIdx.x] += !is_one;
        tid+=blockDim.x;
    }
    __syncthreads();
    int last_zero_count = zero_count[blockDim.x-1];
    cuAccumulateIndex(zero_count, blockDim.x);
    cuAccumulateIndex(one_count, blockDim.x);
    __syncthreads();

    tid = threadIdx.x;
    zero = zero_count[threadIdx.x];
    one = one_count[threadIdx.x];
    while(tid<n)
    {
        if (!partition[tid])
        {
            data[zero]=data_copy[tid];
            zero++;
        }else
        {
            data[n-one-1]=data_copy[tid];
            one++;
        }
        tid+=blockDim.x;
    }
    return zero_count[blockDim.x-1]+last_zero_count;
}

__device__ void cuRadixSelect(Point *data, Point *data_copy, unsigned int m, unsigned int n, int *partition, int dir, Point *result)
{
    __shared__ int one_count[1025];
    __shared__ int zeros_count[1025];

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

        cut = cuPartition(data+l, data_copy, u-l, partition, one_count, zeros_count, bit++, dir);
        __syncthreads();
        if ((l+cut) <= m)
        {
            l +=cut;
        }
        else
        {
            u -=u-cut-l;
        }
    }while ((u > l) && (bit<32));
    if (tid == 0)
    {
        *result = data[m];
    }
}

__global__ void cuRadixSelectGlobal(Point *data, Point *data_copy, unsigned int m, unsigned int n, int *partition, int dir, Point *result)
{
  cuRadixSelect(data, data_copy, m, n, partition, dir, result);
}



__device__ __host__
unsigned int nextPowerOf2(unsigned int x)
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
bool isPowTwo(unsigned int x)
{
  return ((x&(x-1))==0);
}

__device__ __host__
unsigned int prevPowerOf2(unsigned int n){
    if (isPowTwo(n))
    {
        return n;
    }
    n = nextPowerOf2(n);
    return n >>=1;

}


void h_print_matrix(Point* points, int n){
    if (debug)
    {
        printf("#################\n");
        for (int i = 0; i < n; ++i)
        {
            printf("i = %2d:   ", i );
            for (int j = 0; j < 3; ++j)
            {
                printf(  " %5.0f ", points[i].p[j]);
            }
            printf("\n");
        }
    }
}

__device__
void printMatrix(Point* points, int n, int offset){
#if __CUDA_ARCH__>=200
    if (debug)
    {
        __syncthreads();
        if (threadIdx.x ==0 && blockIdx.x ==0)
        {
            printf("####################\n");
            printf("block = %d, blockOffset = %d\n", blockIdx.x, offset );
            for (int i = 0; i < n; ++i)
            {
                printf("i = %3d", i );
                for (int j = 0; j < 3; ++j)
                {
                    printf(  " %3.1f ", points[i].p[j]);
                }
                printf("\n",1 );
            }
            printf("\n####################\n");
        }
        __syncthreads();
    }
#endif
}

__global__
void cuBalanceBranch(Point* points, Point* swap, int *partition, int n, int p, int dir){

    int blockoffset, bid;
    Point result[1];
    bid = blockIdx.x;
    while(bid < p){
        blockoffset = n * bid;
        cuRadixSelect(points+blockoffset, swap+blockoffset, n/2, n, partition+blockoffset, dir, result);
        printMatrix(points, n, blockoffset);
        bid += gridDim.x;
    }

}

void getThreadAndBlockCount(int n, int p, int &blocks, int &threads)
{
    n = n/p;
    n = prevPowerOf2(n/2);
    blocks = min(MAX_BLOCK_DIM_SIZE, p);
    blocks = max(1, blocks);
    threads = min(THREADS_PER_BLOCK, n);
    threads = max(1, threads);
}

void build_kd_tree(Point *h_points, int n)
{

    h_print_matrix(h_points, n);

    Point *d_points, *d_swap;
    int p, h, i, numBlocks, numThreads;
    int *d_partition;

    checkCudaErrors(
        cudaMalloc(&d_partition, n*sizeof(int)));

    checkCudaErrors(
        cudaMalloc(&d_points, n*sizeof(Point)));

    checkCudaErrors(
        cudaMalloc(&d_swap, n*sizeof(Point)));

    checkCudaErrors(
        cudaMemcpy(d_points, h_points, n*sizeof(Point), cudaMemcpyHostToDevice));

    h = ceil(log2((float)n + 1) - 1);
    p = 1;
    for (i = 0; i < h; i++)
    {
        getThreadAndBlockCount(n, p, numBlocks, numThreads);
        debugf("n = %d, p = %d, numblosck = %d, numThread =%d\n", n/p, p, numBlocks, numThreads );
        cuBalanceBranch<<<numBlocks,numThreads>>>(d_points, d_swap, d_partition, n/p, p, i%3);
        p <<=1;
    }

    checkCudaErrors(
        cudaMemcpy(h_points, d_points, n*sizeof(Point), cudaMemcpyDeviceToHost));

    h_print_matrix(h_points, n);

    checkCudaErrors(cudaFree(d_points));
    checkCudaErrors(cudaFree(d_swap));
    checkCudaErrors(cudaFree(d_partition));
}



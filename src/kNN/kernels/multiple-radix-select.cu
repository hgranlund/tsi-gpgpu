#include "multiple-radix-select.cuh"
#include "common.cuh"
#include <stdio.h>

#include <helper_cuda.h>

#define checkCudaErrors(val)           check ( (val), #val, __FILE__, __LINE__ )

# define debug 0
__device__
void printIntArray__(int* l, int n, char *s){
  int i;
  if (debug && threadIdx.x == 0)
  {
    printf("%s: ", s);
    printf("[%d", l[0] );
      for (i = 1; i < n; ++i)
      {
        printf(", %d", l[i] );
      }
      printf("]\n");
    }
  }


  __device__  void cuAccumulateIndex(int *list, int n)
  {
    int i, j, temp, temp_index,
    tid = threadIdx.x;
    if (tid == blockDim.x-1)
    {
      list[-1] =0;
    }
    for ( i = 2; i <= n; i<<=1)
    {
      __syncthreads();
      temp_index = tid * i + i/2 -1;
      if (temp_index+i/2 <n)
      {
        temp = list[temp_index];
        for (j = 1; j <= i/2; ++j)
        {
          list[temp_index + j]+=temp;
        }
      }
    }
  }


  __device__ int cuSumReduce(int *list, int n)
  {
    int half = n/2;
    int tid = threadIdx.x;
    while(half > 0)
    {
      if (tid<half)
      {
        list[tid] += list[tid+half];
      }
      half = half/2;
      __syncthreads();
    }
    return list[0];
  }

  __device__ void cuPartitionSwap(Point *data, Point *swap, unsigned int n, int *partition, int *zero_count, int *one_count, Point median, int dir)
  {
   unsigned int
   tid = threadIdx.x,
   is_bigger,
   big,
   less;
   zero_count++;
   one_count++;
   zero_count[threadIdx.x] = 0;
   one_count[threadIdx.x] = 0;

   while(tid < n)
   {
    swap[tid]=data[tid];
    is_bigger = partition[tid]= (bool)(data[tid].p[dir] > median.p[dir]);
    one_count[threadIdx.x] += is_bigger;
    zero_count[threadIdx.x] += !is_bigger;
    tid+=blockDim.x;
  }
  __syncthreads();
  cuAccumulateIndex(zero_count, blockDim.x);
  cuAccumulateIndex(one_count, blockDim.x);
  tid = threadIdx.x;
  __syncthreads();
  one_count--;
  zero_count--;
  less = zero_count[threadIdx.x];
  big = one_count[threadIdx.x];
  while(tid<n)
  {
    if (!partition[tid])
    {
     data[less]=swap[tid];
     less++;
   }else
   {
     data[n-big-1]=swap[tid];
     big++;
   }
   tid+=blockDim.x;
 }
}

__device__ unsigned int cuPartition(Point *data, unsigned int n, int *partition, int *zero_count, int last, unsigned int bit, int dir)
{
	unsigned int
	tid = threadIdx.x,
	is_one,
	radix = (1 << 31-bit);
	zero_count[threadIdx.x] = 0;
	while(tid < n)
	{
		if (partition[tid] == last)
		{
			is_one = partition[tid] = (bool)((*(int*)&(data[tid].p[dir]))&radix);
      zero_count[threadIdx.x] += !is_one;
    }else{
     partition[tid] = 2;
   }
   tid+=blockDim.x;
 }
 __syncthreads();
 return cuSumReduce(zero_count, blockDim.x);
}

__device__ void cuRadixSelect(Point *data, Point *data_copy, unsigned int m, unsigned int n, int *partition, int dir)
{
	__shared__ int one_count[1025];
	__shared__ int zeros_count[1025];
  __shared__ Point median;


  int l=0,
  u = n,
  m_u = ceil((float)n/2),
  cut=0,
  bit = 0,
  last = 2,
  tid = threadIdx.x;
  while(tid < n)
  {
    partition[tid] = last;
    tid+=blockDim.x;
  }

  tid = threadIdx.x;
  do {
    __syncthreads();
    cut = cuPartition(data, n, partition, zeros_count, last, bit++, dir);
    if ((u-cut) >= (m_u))
    {
      u = u-cut;
      last = 1;
    }
    else
    {
      l =u-cut;
      last = 0;
    }
  }while (((u-l)>1) && (bit<32));

  tid = threadIdx.x;

  __syncthreads();
  while(tid < n)
  {
    if (partition[tid] == last)
    {
     median = data[tid];
     data[tid]=data[0], data[0] = median;
   }
   tid+=blockDim.x;
 }
 __syncthreads();
 cuPartitionSwap(data+1, data_copy, n-1, partition, one_count, zeros_count, median, dir);
 __syncthreads();
 if (threadIdx.x == 0)
 {
   median = data[m];
   data[m]=data[0], data[0] = median;
 }
}

__global__
void cuBalanceBranch(Point* points, Point* swap, int *partition, int n, int p, int dir){

	int blockoffset, bid;
	bid = blockIdx.x;
	while(bid < p){
		blockoffset = n * bid;
		cuRadixSelect(points+blockoffset, swap+blockoffset, n/2, n, partition+blockoffset, dir);
		bid += gridDim.x;
	}
}

//For testing - One cannot import a __device__ kernel
__global__ void cuRadixSelectGlobal(Point *data, Point *data_copy, unsigned int m, unsigned int n, int *partition, int dir)
{
  cuRadixSelect(data, data_copy, m, n, partition, dir);
}



void getThreadAndBlockCountMulRadix(int n, int p, int &blocks, int &threads)
{
  n = n/p;
  n--;
  n = prevPowTwo(n/2);
  blocks = min(MAX_BLOCK_DIM_SIZE, p);
  blocks = max(1, blocks);
  threads = min(THREADS_PER_BLOCK, n);
  threads = max(1, threads);
}

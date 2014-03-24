#include "radix-select.cuh"
#include <stdio.h>

#include <helper_cuda.h>

#define checkCudaErrors(val)           check ( (val), #val, __FILE__, __LINE__ )

# define debug 0
#define FILE (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define debugf(fmt, ...) if(debug)printf("%s:%d: " fmt, FILE, __LINE__, __VA_ARGS__);

__device__
void d_printIntArray___(int* l, int n, char *s){
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
    __syncthreads();
  }

  void h_printIntArray___(int* l, int n, char *s)
  {
    int i;
    if (debug)
    {
      printf("%s: ", s);
      printf("[%d", l[0]);
        for (i = 1; i < n; ++i)
        {
          printf(", %d", l[i] );
        }
        printf("]\n");
      }
    }


    __device__  void cuAccumulateIndex_(int *list, int n)
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


    __device__ void cuSumReduce_(int *list, int n)
    {
      unsigned int tid = threadIdx.x;

      if (n >= 1024)
      {
        if (tid < 512)
        {
          list[tid] += list[tid + 512];
        }
        __syncthreads();
      }

      if (n >= 512)
      {
        if (tid < 256)
        {
          list[tid] += list[tid + 256];
        }
        __syncthreads();
      }

      if (n >= 256)
      {
        if (tid < 128)
        {
          list[tid] += list[tid + 128];
        }
        __syncthreads();
      }

      if (n >= 128)
      {
        if (tid <  64)
        {
          list[tid] += list[tid +  64];
        }
        __syncthreads();
      }

      if (tid < 32)
      {
        volatile int *smem = list;

        if (n >=  64)
        {
          smem[tid] += smem[tid + 32];
        }

        if (n >=  32)
        {
          smem[tid] += smem[tid + 16];
        }

        if (n >=  16)
        {
          smem[tid] += smem[tid +  8];
        }

        if (n >=   8)
        {
          smem[tid] += smem[tid +  4];
        }

        if (n >=   4)
        {
          smem[tid] += smem[tid +  2];
        }

        if (n >=   2)
        {
          smem[tid] += smem[tid +  1];
        }
      }
    }

    __global__ void cuPartitionSwap(Point *points, Point *swap, int n, int *partition, int last, int dir)
    {
      __shared__ int ones[1025];
      __shared__ int zeros[1025];
      __shared__ Point median;

      int
      tid = threadIdx.x,
      is_bigger,
      big,
      *zero_count = ones,
      *one_count = zeros,
      less;

      zero_count++;
      one_count++;
      zero_count[threadIdx.x] = 0;
      one_count[threadIdx.x] = 0;

      while(tid < n)
      {
        if (partition[tid] == last)
        {
          median = points[tid];
          points[tid] = points[0], points[0] = median;
          // printf("tid= %d, median = %3.1f \n",tid, median.p[0]);
        }
        tid+=blockDim.x;
      }
      points++;
      n--;
      __syncthreads();
      tid = threadIdx.x;
      while(tid < n)
      {
        swap[tid]=points[tid];
        is_bigger = partition[tid]= (bool)(points[tid].p[dir] > median.p[dir]);
        one_count[threadIdx.x] += is_bigger;
        zero_count[threadIdx.x] += !is_bigger;
        tid+=blockDim.x;
      }
      __syncthreads();
      cuAccumulateIndex_(zero_count, blockDim.x);
      cuAccumulateIndex_(one_count, blockDim.x);
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
         points[less]=swap[tid];
         less++;
       }else
       {
         points[n-big-1]=swap[tid];
         big++;
       }
       tid+=blockDim.x;
     }
     __syncthreads();
     if (threadIdx.x == 0)
     {
      n++;
      points--;
      points[0] = points[n/2], points[n/2] =median;
    }
  }

  __global__ void cuPartitionStep(Point *data, unsigned int n, int *partition, int *zeros_count_block, int last, unsigned int bit, int dir)
  {
    __shared__ int zero_count[1024];

    int
    tid = threadIdx.x,
    is_one,
    block_offset,
    rest = n % gridDim.x,
    block_step,
    radix = (1 << 31-bit);
    zero_count[threadIdx.x] = 0;
    block_step = n/gridDim.x;
    n = block_step;
    block_offset = block_step * blockIdx.x;
    if (rest >= gridDim.x-blockIdx.x)
    {
      block_offset += rest - (gridDim.x - blockIdx.x);
      n++;
    }
    data += block_offset;
    partition += block_offset;
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
   cuSumReduce_(zero_count, blockDim.x);
   if (threadIdx.x == 0)
   {
     zeros_count_block[blockIdx.x] = zero_count[0];
   }
 }


 int sumReduce(int *list, int n){
  int i, sum = 0;
  for (i = 0; i < n; ++i)
  {
    sum+=list[i];
  }
  return sum;
}

void getThreadAndBlockCountPartition(int n, int &blocks, int &threads)
{
  threads = 512;
  blocks = n/threads/2;
  blocks = max(1, blocks);
  blocks = min(MAX_BLOCK_DIM_SIZE, blocks);
}

__global__ void fillArray(int *array, int value, int n)
{
  int tid = threadIdx.x;
  while(tid <n)
  {
    array[tid]=value;
    tid += blockDim.x;
  }
}

void radixSelectAndPartition(Point* points, Point* swap, int *partition, int m, int n, int dir)
{
  int numBlocks, numThreads,
  l=0,
  u = n,
  m_u = ceil((float)n/2),
  cut=0,
  bit = 0,
  last = 2,
  *h_zeros_count_block,
  *d_zeros_count_block;

  numThreads = min(n, THREADS_PER_BLOCK);
  fillArray<<<1,numThreads>>>(partition, 2, n);
  getThreadAndBlockCountPartition(n,numBlocks, numThreads);
  debugf("blocks = %d, threads = %d\n", numBlocks, numThreads);
  h_zeros_count_block = (int*) malloc(numBlocks*sizeof(int));
  checkCudaErrors(
    cudaMalloc((void **) &d_zeros_count_block, numBlocks*sizeof(int)));

  do {
    cuPartitionStep<<<numBlocks, numThreads>>>(points, n, partition, d_zeros_count_block, last, bit++, dir);
    cudaMemcpy(h_zeros_count_block, d_zeros_count_block, numBlocks*sizeof(int), cudaMemcpyDeviceToHost);
    cut = sumReduce(h_zeros_count_block, numBlocks);
    if ((l+cut) > m_u)
    {
      u = l+cut;
      last = 0;
    }
    else
    {
      l = l+cut;
      last = 1;
    }
  }while (((u-l)>1) && (bit<32));

  cuPartitionSwap<<<1,numThreads>>>(points, swap, n, partition, last, dir);

  checkCudaErrors(
    cudaFree(d_zeros_count_block));
}





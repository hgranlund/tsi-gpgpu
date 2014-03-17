#include "radix-select.cuh"
#include <stdio.h>

#include <helper_cuda.h>

#define checkCudaErrors(val)           check ( (val), #val, __FILE__, __LINE__ )

# define debug 1
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

  void h_printIntArray___(int* l, int n, char *s){
    int i;
    if (debug)
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


    __device__ int cuSumReduce_(int *list, int n)
    {
      int half = n/2;
      int tid = threadIdx.x;
      while(tid<half && half > 0)
      {
        list[tid] += list[tid+half];
        half = half/2;
      }
      __syncthreads();
      return list[0];
    }

    __global__ void cuPartitionSwap(Point *points, Point *swap, int n, int *partition, int last, int dir)
    {
      __shared__ int ones[513];
      __shared__ int zeros[513];
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
    block_step,
    radix = (1 << 31-bit);
    zero_count[threadIdx.x] = 0;
    block_step = n/gridDim.x;
    n = block_step;
    data += block_step * blockIdx.x;
    partition += block_step * blockIdx.x;
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
   zeros_count_block[blockIdx.x] = cuSumReduce_(zero_count, blockDim.x);
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
  blocks = n/threads;
  blocks = max(1, blocks);
  // threads = max(1, threads);
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
  *h_zeros,
  *d_zeros_count_block;



  getThreadAndBlockCountPartition(n,numBlocks, numThreads);
  numThreads=2;
  h_zeros_count_block = (int*) malloc(numBlocks*sizeof(int));
  h_zeros = (int*) malloc(numBlocks*sizeof(int));
  for (int i = 0; i < numBlocks; ++i)
  {
    h_zeros[i] = 0;
  }
  checkCudaErrors(
    cudaMalloc((void **) &d_zeros_count_block, numBlocks*sizeof(int)));

  fillArray<<<1,512>>>(partition, 2, n);

  do {
    cudaMemcpy(d_zeros_count_block, h_zeros, numBlocks*sizeof(int), cudaMemcpyHostToDevice);
    cuPartitionStep<<<numBlocks, numThreads>>>(points, n, partition, d_zeros_count_block, last, bit++, dir);
    cudaMemcpy(h_zeros_count_block, d_zeros_count_block, numBlocks*sizeof(int), cudaMemcpyDeviceToHost);
    cut = sumReduce(h_zeros_count_block, numBlocks);
    if ((u-cut) >= m_u)
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

  cuPartitionSwap<<<1,numThreads>>>(points, swap, n, partition, last, dir);

  checkCudaErrors(
    cudaFree(d_zeros_count_block));

 //  __syncthreads();
 //  while(tid < n)
 //  {
 //    if (partition[tid] == last)
 //    {
 //     median = data[tid];
 //     data[tid]=data[0], data[0] = median;

 //   }
 //   tid+=blockDim.x;
 // }
 // __syncthreads();
 // cuPartitionSwap(data+1, data_copy, n-1, partition, one_count, zeros_count, median, dir);
 // __syncthreads();
 // if (threadIdx.x == 0)
 // {
 //   median = data[m];
 //   data[m]=data[0], data[0] = median;
 // }
}





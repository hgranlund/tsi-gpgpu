#include "reduction-mod.cuh"
#include "cuda.h"
#include "stdio.h"
#include "helper_cuda.h"

#define CUDART_INF_F  __int_as_float(0x7f800000)
#define THREADS_PER_BLOCK 512U
#define MAX_BLOCK_DIM_SIZE 65535u


bool isPow2(unsigned int x)
{
  return ((x&(x-1))==0);
}

__device__ void cuMinR(Distance &distA, Distance &distB, unsigned int &min_index, unsigned int index, unsigned int dir)
{
  if ((distA.value  >= distB.value) == dir)
  {
    distA  = distB;
    min_index = index;
  }
}

unsigned int nextPow2(unsigned int x)
{
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return ++x;
}

////////////////////////////////////////////////////////////////////////////////
// Compute the number of threads and blocks to use
// For the kernels >= 3, we set threads / block to the minimum of maxThreads and
// n/2. For kernels < 3, we set to the minimum of maxThreads and n.  For kernel
// 6, we observe the maximum specified number of blocks, because each thread in
// that kernel can process a variable number of elements.
////////////////////////////////////////////////////////////////////////////////
void getNumBlocksAndThreads(int n, int maxBlocks, int maxThreads, int &blocks, int &threads)
{

    //get device capability, to avoid block/grid size excceed the upbound
  cudaDeviceProp prop;
  int device;
  checkCudaErrors(cudaGetDevice(&device));
  checkCudaErrors(cudaGetDeviceProperties(&prop, device));


  threads = (n < maxThreads*2) ? nextPow2((n + 1)/ 2) : maxThreads;
  blocks = (n + (threads * 2 - 1)) / (threads * 2);


  if ((float)threads*blocks > (float)prop.maxGridSize[0] * prop.maxThreadsPerBlock)
  {
    printf("n is too large, please choose a smaller number!\n");
  }

  if (blocks > prop.maxGridSize[0])
  {
    printf("Grid size <%d> excceeds the device capability <%d>, set block size as %d (original %d)\n",
     blocks, prop.maxGridSize[0], threads*2, threads);

    blocks /= 2;
    threads *= 2;
  }

  blocks = min(maxBlocks, blocks);

}


/*
    This version adds multiple elements per thread sequentially.  This reduces the overall
    cost of the algorithm while keeping the work complexity O(n) and the step complexity O(log n).
    (Brent's Theorem optimization)

    Note, this kernel needs a minimum of 64*sizeof(T) bytes of shared memory.
    In other words if blockSize <= 32, allocate 64*sizeof(T) bytes.
    If blockSize > 32, allocate blockSize*sizeof(T) bytes.
*/
template <unsigned int blockSize, bool nIsPow2>
    __global__ void reduce6(Distance *g_dist, unsigned int n)
    {
      __shared__ Distance s_dist[blockSize];
      __shared__ int s_ind[blockSize];
      unsigned int dir = 1;

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
      unsigned int tid = threadIdx.x;
      unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
      unsigned int gridSize = blockSize*2*gridDim.x;

      Distance min_dist = {1,CUDART_INF_F};
      unsigned int min_index = 0;
    // we reduce multiple elements per thread.  The number is determin_listed by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread

      while (i < n)
      {
        cuMinR(min_dist,  g_dist[i] ,min_index,i,dir);
        if (nIsPow2 || i + blockSize < n){
          cuMinR(min_dist,  g_dist[i+blockSize], min_index, i+blockSize ,dir);
        }
        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        i += gridSize;
      }

      // each thread puts its local min into shared memory

      s_dist[tid] = min_dist;
      s_ind[tid] = min_index;

      __syncthreads();


    // do reduction in shared mem
      if (blockSize >= 512)
      {
        if (tid < 256)
        {
          cuMinR(min_dist,  s_dist[tid+256], min_index, s_ind[tid+256] ,dir);
          s_dist[tid] = min_dist;
          s_ind[tid] = min_index;
        }
        __syncthreads();
      }

      if (blockSize >= 256)
      {
        if (tid < 128)
        {
          cuMinR(min_dist,  s_dist[tid+128], min_index, s_ind[tid+128] ,dir);
          s_ind[tid] = min_index;
          s_dist[tid] = min_dist;
        }
        __syncthreads();
      }

      if (blockSize >= 128)
      {
        if (tid <  64)
        {
          cuMinR(min_dist,  s_dist[tid+64], min_index, s_ind[tid+64] ,dir);
          s_ind[tid] = min_index;
          s_dist[tid] = min_dist;
        }
        __syncthreads();
      }

      if (tid < 32)
      {

        // now that we are using warp-synchronous programmin_listg (below)
        // we need to declare our shared memory volatile so that the compiler
        // doesn't reorder stores to it and induce incorrect behavior.
        volatile  int *v_ind = s_ind;
        volatile  Distance *v_dist = s_dist;

        if (blockSize >=  64)
        {
          if ((min_dist.value >= v_dist[tid+32].value)==dir)
          {
            min_dist =v_dist[tid]= v_dist[tid+32];
            min_index =v_ind[tid]= v_ind[tid+32];
          }

        }

        if (blockSize >=  32)
        {
          if ((min_dist.value >= v_dist[tid+16].value)==dir)
          {
            min_dist =v_dist[tid]= v_dist[tid+16];
            min_index =v_ind[tid]= v_ind[tid+16];
          }
        }

        if (blockSize >=  16)
        {
          if ((min_dist.value >= v_dist[tid+8].value)==dir)
          {
            min_dist =v_dist[tid]= v_dist[tid+8];
            min_index =v_ind[tid]= v_ind[tid+8];
          }

        }


        if (blockSize >=   8)
        {
          if ((min_dist.value >= v_dist[tid+4].value)==dir)
          {
            min_dist =v_dist[tid]= v_dist[tid+4];
            min_index =v_ind[tid]= v_ind[tid+4];
          }

        }

        if (blockSize >=   4)
        {
          if ((min_dist.value >= v_dist[tid+2].value)==dir)
          {
            min_dist =v_dist[tid]= v_dist[tid+2];
            min_index =v_ind[tid]= v_ind[tid+2];
          }

        }


        if (blockSize >=   2)
        {
          if ((min_dist.value >= v_dist[tid+1].value)==dir)
          {
            min_dist =v_dist[tid]= v_dist[tid+1];
            min_index =v_ind[tid]= v_ind[tid+1];
          }

        }
      }

    // Swap smallest value to start of g_dist array
      if (tid == 0){
        i = blockIdx.x;
        min_dist = g_dist[i];
        g_dist[i]=g_dist[s_ind[tid]];
        g_dist[s_ind[tid]]=min_dist;
      }
    }


////////////////////////////////////////////////////////////////////////////////
// Wrapper function for kernel launch
////////////////////////////////////////////////////////////////////////////////
    void reduce(int size, int threads, int blocks, Distance *g_dist)
    {
      dim3 dimBlock(threads, 1, 1);
      dim3 dimGrid(blocks, 1, 1);

// when there is only one warp per block, we need to allocate two warps
// worth of shared memory so that we don't index shared memory out of bounds
      int smemSize = (threads <= 32) ? 2 * threads * (sizeof(Distance) + sizeof(int))  : threads * (sizeof(Distance) + sizeof(int));
      if (isPow2(size))
      {
        switch (threads)
        {
          case 512:
          reduce6< 512, true><<< dimGrid, dimBlock, smemSize >>>(g_dist, size); break;
          case 256:
          reduce6< 256, true><<< dimGrid, dimBlock, smemSize >>>(g_dist, size); break;
          case 128:
          reduce6< 128, true><<< dimGrid, dimBlock, smemSize >>>(g_dist, size); break;
          case 64:
          reduce6<  64, true><<< dimGrid, dimBlock, smemSize >>>(g_dist, size); break;
          case 32:
          reduce6<  32, true><<< dimGrid, dimBlock, smemSize >>>(g_dist, size); break;
          case 16:
          reduce6<  16, true><<< dimGrid, dimBlock, smemSize >>>(g_dist, size); break;
          case  8:
          reduce6<   8, true><<< dimGrid, dimBlock, smemSize >>>(g_dist, size); break;
          case  4:
          reduce6<   4, true><<< dimGrid, dimBlock, smemSize >>>(g_dist, size); break;
          case  2:
          reduce6<   2, true><<< dimGrid, dimBlock, smemSize >>>(g_dist, size); break;
          case  1:
          reduce6<   1, true><<< dimGrid, dimBlock, smemSize >>>(g_dist, size); break;
        }
      }
      else
      {
        switch (threads)
        {
          case 512:
          reduce6< 512, false><<< dimGrid, dimBlock, smemSize >>>(g_dist, size); break;
          case 256:
          reduce6< 256, false><<< dimGrid, dimBlock, smemSize >>>(g_dist, size); break;
          case 128:
          reduce6< 128, false><<< dimGrid, dimBlock, smemSize >>>(g_dist, size); break;
          case 64:
          reduce6<  64, false><<< dimGrid, dimBlock, smemSize >>>(g_dist, size); break;
          case 32:
          reduce6<  32, false><<< dimGrid, dimBlock, smemSize >>>(g_dist, size); break;
          case 16:
          reduce6<  16, false><<< dimGrid, dimBlock, smemSize >>>(g_dist, size); break;
          case  8:
          reduce6<   8, false><<< dimGrid, dimBlock, smemSize >>>(g_dist, size); break;
          case  4:
          reduce6<   4, false><<< dimGrid, dimBlock, smemSize >>>(g_dist, size); break;
          case  2:
          reduce6<   2, false><<< dimGrid, dimBlock, smemSize >>>(g_dist, size); break;
          case  1:
          reduce6<   1, false><<< dimGrid, dimBlock, smemSize >>>(g_dist, size); break;
        }
      }
    }


    void dist_min_reduce(Distance *g_dist, unsigned int n)
    {
      int numBlocks = 0;
      int numThreads = 0;
      getNumBlocksAndThreads(n, MAX_BLOCK_DIM_SIZE, THREADS_PER_BLOCK, numBlocks, numThreads);
      reduce(n, numThreads, numBlocks, g_dist);
      n=numBlocks;
      if (n >1)
      {
        getNumBlocksAndThreads(n, MAX_BLOCK_DIM_SIZE, THREADS_PER_BLOCK, numBlocks, numThreads);
        reduce(n, numThreads,1,g_dist);
      }
    }


#include "reduction-mod.cuh"
#include "cuda.h"
#include "stdio.h"
#include "helper_cuda.h"

#define CUDART_INF_F  __int_as_float(0x7f800000)
#define THREADS_PER_BLOCK 512U
#define MAX_BLOCK_DIM_SIZE 65535U


bool isPow2(int x)
{
    return ((x & (x - 1)) == 0);
}

__device__ void cuMinR(Distance &distA, Distance &distB, int &min_index, int index, int dir)
{
    if ((distA.value  >= distB.value) == dir)
    {
        distA  = distB;
        min_index = index;
    }
}

int nextPow2(int x)
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

void getNumBlocksAndThreads(int n, int maxBlocks, int maxThreads, int &blocks, int &threads)
{

    threads = (n < maxThreads * 2) ? nextPow2((n + 1) / 2) : maxThreads;
    blocks = (n + (threads * 2 - 1)) / (threads * 2);
    blocks = min(maxBlocks, blocks);
}

template <int blockSize, bool nIsPow2>
__global__ void cuReduce(Distance *g_dist, int n)
{
    __shared__ Distance s_dist[blockSize];
    __shared__ int s_ind[blockSize];
    int dir = 1;

    Distance min_dist = {1, CUDART_INF_F};
    int min_index = 0;

    int tid = threadIdx.x;
    int i = blockIdx.x * blockSize * 2 + threadIdx.x;
    int gridSize = blockSize * 2 * gridDim.x;

    while (i < n)
    {
        cuMinR(min_dist,  g_dist[i] , min_index, i, dir);
        if (nIsPow2 || i + blockSize < n)
        {
            cuMinR(min_dist,  g_dist[i + blockSize], min_index, i + blockSize , dir);
        }
        i += gridSize;
    }

    s_dist[tid] = min_dist;
    s_ind[tid] = min_index;

    __syncthreads();

    if (blockSize >= 512)
    {
        if (tid < 256)
        {
            cuMinR(min_dist,  s_dist[tid + 256], min_index, s_ind[tid + 256] , dir);
            s_dist[tid] = min_dist;
            s_ind[tid] = min_index;
        }
        __syncthreads();
    }

    if (blockSize >= 256)
    {
        if (tid < 128)
        {
            cuMinR(min_dist,  s_dist[tid + 128], min_index, s_ind[tid + 128] , dir);
            s_ind[tid] = min_index;
            s_dist[tid] = min_dist;
        }
        __syncthreads();
    }

    if (blockSize >= 128)
    {
        if (tid <  64)
        {
            cuMinR(min_dist,  s_dist[tid + 64], min_index, s_ind[tid + 64] , dir);
            s_ind[tid] = min_index;
            s_dist[tid] = min_dist;
        }
        __syncthreads();
    }

    if (tid < 32)
    {

        volatile  int *v_ind = s_ind;
        volatile  Distance *v_dist = s_dist;

        if (blockSize >=  64)
        {
            if ((min_dist.value >= v_dist[tid + 32].value) == dir)
            {
                min_dist = v_dist[tid] = v_dist[tid + 32];
                min_index = v_ind[tid] = v_ind[tid + 32];
            }
        }

        if (blockSize >=  32)
        {
            if ((min_dist.value >= v_dist[tid + 16].value) == dir)
            {
                min_dist = v_dist[tid] = v_dist[tid + 16];
                min_index = v_ind[tid] = v_ind[tid + 16];
            }
        }

        if (blockSize >=  16)
        {
            if ((min_dist.value >= v_dist[tid + 8].value) == dir)
            {
                min_dist = v_dist[tid] = v_dist[tid + 8];
                min_index = v_ind[tid] = v_ind[tid + 8];
            }
        }

        if (blockSize >=   8)
        {
            if ((min_dist.value >= v_dist[tid + 4].value) == dir)
            {
                min_dist = v_dist[tid] = v_dist[tid + 4];
                min_index = v_ind[tid] = v_ind[tid + 4];
            }
        }

        if (blockSize >=   4)
        {
            if ((min_dist.value >= v_dist[tid + 2].value) == dir)
            {
                min_dist = v_dist[tid] = v_dist[tid + 2];
                min_index = v_ind[tid] = v_ind[tid + 2];
            }
        }

        if (blockSize >=   2)
        {
            if ((min_dist.value >= v_dist[tid + 1].value) == dir)
            {
                min_dist = v_dist[tid] = v_dist[tid + 1];
                min_index = v_ind[tid] = v_ind[tid + 1];
            }
        }
    }

    if (tid == 0)
    {
        i = blockIdx.x;
        min_dist = g_dist[i];
        g_dist[i] = g_dist[s_ind[tid]];
        g_dist[s_ind[tid]] = min_dist;
    }
}


void reduce(int size, int threads, int blocks, Distance *g_dist)
{
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    int smemSize = (threads <= 32) ? 2 * threads * (sizeof(Distance) + sizeof(int))  : threads * (sizeof(Distance) + sizeof(int));
    if (isPow2(size))
    {
        switch (threads)
        {
        case 512:
            cuReduce< 512, true> <<< dimGrid, dimBlock, smemSize >>>(g_dist, size); break;
        case 256:
            cuReduce< 256, true> <<< dimGrid, dimBlock, smemSize >>>(g_dist, size); break;
        case 128:
            cuReduce< 128, true> <<< dimGrid, dimBlock, smemSize >>>(g_dist, size); break;
        case 64:
            cuReduce<  64, true> <<< dimGrid, dimBlock, smemSize >>>(g_dist, size); break;
        case 32:
            cuReduce<  32, true> <<< dimGrid, dimBlock, smemSize >>>(g_dist, size); break;
        case 16:
            cuReduce<  16, true> <<< dimGrid, dimBlock, smemSize >>>(g_dist, size); break;
        case  8:
            cuReduce<   8, true> <<< dimGrid, dimBlock, smemSize >>>(g_dist, size); break;
        case  4:
            cuReduce<   4, true> <<< dimGrid, dimBlock, smemSize >>>(g_dist, size); break;
        case  2:
            cuReduce<   2, true> <<< dimGrid, dimBlock, smemSize >>>(g_dist, size); break;
        case  1:
            cuReduce<   1, true> <<< dimGrid, dimBlock, smemSize >>>(g_dist, size); break;
        }
    }
    else
    {
        switch (threads)
        {
        case 512:
            cuReduce< 512, false> <<< dimGrid, dimBlock, smemSize >>>(g_dist, size); break;
        case 256:
            cuReduce< 256, false> <<< dimGrid, dimBlock, smemSize >>>(g_dist, size); break;
        case 128:
            cuReduce< 128, false> <<< dimGrid, dimBlock, smemSize >>>(g_dist, size); break;
        case 64:
            cuReduce<  64, false> <<< dimGrid, dimBlock, smemSize >>>(g_dist, size); break;
        case 32:
            cuReduce<  32, false> <<< dimGrid, dimBlock, smemSize >>>(g_dist, size); break;
        case 16:
            cuReduce<  16, false> <<< dimGrid, dimBlock, smemSize >>>(g_dist, size); break;
        case  8:
            cuReduce<   8, false> <<< dimGrid, dimBlock, smemSize >>>(g_dist, size); break;
        case  4:
            cuReduce<   4, false> <<< dimGrid, dimBlock, smemSize >>>(g_dist, size); break;
        case  2:
            cuReduce<   2, false> <<< dimGrid, dimBlock, smemSize >>>(g_dist, size); break;
        case  1:
            cuReduce<   1, false> <<< dimGrid, dimBlock, smemSize >>>(g_dist, size); break;
        }
    }
}


void dist_min_reduce(Distance *g_dist, int n)
{
    int numBlocks = 0;
    int numThreads = 0;
    getNumBlocksAndThreads(n, MAX_BLOCK_DIM_SIZE, THREADS_PER_BLOCK, numBlocks, numThreads);

    reduce(n, numThreads, numBlocks, g_dist);
    n = numBlocks;
    while (n > 1)
    {
        getNumBlocksAndThreads(n, MAX_BLOCK_DIM_SIZE, THREADS_PER_BLOCK, numBlocks, numThreads);
        reduce(n, numThreads, numBlocks, g_dist);
        n = numBlocks;
    }
}


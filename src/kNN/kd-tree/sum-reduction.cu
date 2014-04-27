#include "sum-reduction.cuh"

#define MAX_BLOCK_DIM_SIZE_SUM_REDUCTION 65535U
#define THREADS_PER_BLOCK_SUM_REDUCTION 512U

template <int blockSize>
__global__ void cuSumReduce__(int *list, int n)
{
    __shared__ int s_data[blockSize];

    int tid = threadIdx.x;
    int i = threadIdx.x;
    int gridSize = blockSize * 2;

    int mySum = 0;

    while (i < n)
    {
        mySum += list[i];
        if (i + blockSize < n)
        {
            mySum += list[i + blockSize];
        }

        i += gridSize;
    }

    s_data[tid] = mySum;
    __syncthreads();

    if (blockSize >= 512)
    {
        if (tid < 256)
        {
            s_data[tid] = mySum = mySum + s_data[tid + 256];
        }

        __syncthreads();
    }

    if (blockSize >= 256)
    {
        if (tid < 128)
        {
            s_data[tid] = mySum = mySum + s_data[tid + 128];
        }

        __syncthreads();
    }

    if (blockSize >= 128)
    {
        if (tid <  64)
        {
            s_data[tid] = mySum = mySum + s_data[tid +  64];
        }

        __syncthreads();
    }

    if (tid < 32)
    {
        volatile int *smem = s_data;

        if (blockSize >=  64)
        {
            smem[tid] = mySum = mySum + smem[tid + 32];
        }

        if (blockSize >=  32)
        {
            smem[tid] = mySum = mySum + smem[tid + 16];
        }

        if (blockSize >=  16)
        {
            smem[tid] = mySum = mySum + smem[tid +  8];
        }

        if (blockSize >=   8)
        {
            smem[tid] = mySum = mySum + smem[tid +  4];
        }

        if (blockSize >=   4)
        {
            smem[tid] = mySum = mySum + smem[tid +  2];
        }

        if (blockSize >=   2)
        {
            smem[tid] = mySum = mySum + smem[tid +  1];
        }
    }

    if (threadIdx.x == 0)
    {
        list[0] = s_data[0];
    }
}

int nextPowerOf2__(int x)
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

void sum_reduce(int *d_list, int n)
{
    int threads, blocks = 1;
    threads = max(1, nextPowerOf2__(((n - 1) >> 1))), //threads must be power of 2
    threads = min(threads, THREADS_PER_BLOCK_SUM_REDUCTION);
    int smemSize = (threads <= 32) ? 2 * threads * sizeof(float) : threads * sizeof(float);

    switch (threads)
    {
    case 512:
        cuSumReduce__<512> <<< blocks, threads, smemSize >>>(d_list, n); break;
    case 256:
        cuSumReduce__<256> <<< blocks, threads, smemSize >>>(d_list, n); break;
    case 128:
        cuSumReduce__<128> <<< blocks, threads, smemSize >>>(d_list, n); break;
    case 64:
        cuSumReduce__<64> <<< blocks, threads, smemSize >>>(d_list, n); break;
    case 32:
        cuSumReduce__<32> <<< blocks, threads, smemSize >>>(d_list, n); break;
    case 16:
        cuSumReduce__<16> <<< blocks, threads, smemSize >>>(d_list, n); break;
    case  8:
        cuSumReduce__<8> <<< blocks, threads, smemSize >>>(d_list, n); break;
    case  4:
        cuSumReduce__<4> <<< blocks, threads, smemSize >>>(d_list, n); break;
    case  2:
        cuSumReduce__<2> <<< blocks, threads, smemSize >>>(d_list, n); break;
    case  1:
        cuSumReduce__<1> <<< blocks, threads, smemSize >>>(d_list, n); break;
    }
}


#include "multiple-radix-select.cuh"
#include <stdio.h>

#include <helper_cuda.h>

#define checkCudaErrors(val)           check ( (val), #val, __FILE__, __LINE__ )

# define debug 0
#define FILE (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define debugf(fmt, ...) if(debug)printf("%s:%d: " fmt, FILE, __LINE__, __VA_ARGS__);

int nextPowTwo(int x)
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

bool isPowTwo(int x)
{
    return ((x & (x - 1)) == 0);
}

int prevPowTwo(int n)
{
    if (isPowTwo(n))
    {
        return n;
    }
    n = nextPowTwo(n);
    return n >>= 1;
}

__device__  void cuAccumulateIndex(int *list, int n)
{
    int i, j, temp,
        tid = threadIdx.x;
    if (tid == blockDim.x - 1)
    {
        list[-1] = 0;
    }
    for ( i = 1; i <= n; i <<= 1)
    {
        __syncthreads();
        int temp_index = tid * i * 2  + i - 1;
        if (temp_index + i < n)
        {
            temp = list[temp_index];
            for (j = 1; j <= i; ++j)
            {
                list[temp_index + j] += temp;
            }
        }
    }
}

__device__ void fillArray_(int *partition, int last, int  n)
{
    int tid = threadIdx.x;
    while (tid < n)
    {
        partition[tid] = last;
        tid += blockDim.x;
    }
}

__device__ int cuSumReduce(int *list, int n)
{
    int tid = threadIdx.x;

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
    __syncthreads();
    return list[0];
}

__device__ void cuPartitionSwap(struct Point *data, struct Point *swap, int n, int *partition, int *zero_count, int *one_count, int *median_count, float median_value, int dir)
{
    int tid = threadIdx.x,
        big,
        mid,
        par,
        less;

    float point_difference;
    struct Point point;

    zero_count++;
    one_count++;
    median_count++;
    zero_count[threadIdx.x] = 0;
    one_count[threadIdx.x] = 0;
    median_count[threadIdx.x] = 0;

    while (tid < n)
    {
        point = data[tid];
        swap[tid] = point;
        point_difference = (point.p[dir] - median_value);
        if (point_difference < 0)
        {
            par = -1;
            zero_count[threadIdx.x]++;
        }
        else if (point_difference > 0)
        {
            par = 1;
            one_count[threadIdx.x]++;
        }
        else
        {
            par = 0;
            median_count[threadIdx.x]++;
        }
        partition[tid] = par;
        tid += blockDim.x;
    }

    __syncthreads();
    cuAccumulateIndex(zero_count, blockDim.x);
    cuAccumulateIndex(one_count, blockDim.x);
    cuAccumulateIndex(median_count, blockDim.x);
    __syncthreads();

    tid = threadIdx.x;
    one_count--;
    zero_count--;
    median_count--;
    less = zero_count[threadIdx.x];
    big = one_count[threadIdx.x];
    mid = zero_count[blockDim.x] +  median_count[threadIdx.x];

    while (tid < n)
    {
        if (partition[tid] < 0)
        {
            data[less] = swap[tid];
            less++;
        }
        else if (partition[tid] > 0)
        {
            data[n - big - 1] = swap[tid];
            big++;
        }
        else
        {
            data[mid] = swap[tid];
            mid++;
        }
        tid += blockDim.x;
    }
}

__device__ int cuPartition(struct Point *data, int n, int *partition, int *zero_count, int last, int bit, int dir)
{
    int is_one,
        tid = threadIdx.x,
        radix = (1 << 31 - bit);

    zero_count[threadIdx.x] = 0;

    while (tid < n)
    {
        if (partition[tid] == last)
        {
            is_one = partition[tid] = (bool)((*(int *) & (data[tid].p[dir]))&radix);
            zero_count[threadIdx.x] += !is_one;
        }
        else
        {
            partition[tid] = 2;
        }
        tid += blockDim.x;
    }

    __syncthreads();
    return cuSumReduce(zero_count, blockDim.x);
}

__device__ void cuRadixSelect(struct Point *data, struct Point *data_copy, int n, int *partition, int dir)
{
    __shared__ int one_count[1025];
    __shared__ int zeros_count[1025];
    __shared__ int medians_count[1025];
    __shared__ float median_value;

    int l = 0,
        bit = 0,
        last = 2,
        u = n,
        m_u = ceil((float)n / 2),
        tid = threadIdx.x;

    fillArray_(partition, last, n);

    do
    {
        __syncthreads();
        int cut = cuPartition(data, n, partition, zeros_count, last, bit++, dir);

        if ((u - cut) >= (m_u))
        {
            u = u - cut;
            last = 1;
        }
        else
        {
            l = u - cut;
            last = 0;
        }
    }
    while (((u - l) > 1) && (bit < 32));

    tid = threadIdx.x;

    while (tid < n)
    {
        if (partition[tid] == last)
        {
            median_value = data[tid].p[dir];
        }
        tid += blockDim.x;
    }
    __syncthreads();

    cuPartitionSwap(data, data_copy, n, partition, one_count, zeros_count, medians_count, median_value, dir);
}

__global__
void cuBalanceBranch(struct Point *points, struct Point *swap, int *partition, int *steps, int p, int dir)
{

    int blockoffset,
        n,
        bid = blockIdx.x;

    while (bid < p)
    {
        blockoffset = steps[bid * 2];
        n = steps[bid * 2 + 1] - blockoffset;
        cuRadixSelect(points + blockoffset, swap + blockoffset, n, partition + blockoffset, dir);
        bid += gridDim.x;
    }
}

void getThreadAndBlockCountMulRadix(int n, int p, int &blocks, int &threads)
{
    threads = n - 1;
    threads = prevPowTwo(threads / 4);
    blocks = min(MAX_BLOCK_DIM_SIZE_MULTI_RADIX, p);
    blocks = max(1, blocks);
    threads = min(THREADS_PER_BLOCK_MULTI_RADIX, threads);
    threads = max(128, threads);
    debugf("N =%d, p = %d, blocks = %d, threads = %d\n", n, p, blocks, threads);
}

void  multiRadixSelectAndPartition(struct Point *d_data, struct Point *d_data_copy, int *d_partition, int *d_steps, int n, int p,  int dir)
{
    int numBlocks, numThreads;
    getThreadAndBlockCountMulRadix(n, p, numBlocks, numThreads);
    cuBalanceBranch <<< numBlocks, numThreads>>>(d_data, d_data_copy, d_partition, d_steps, p, dir);
}

//For testing - One cannot import a __device__ kernel
__global__ void cuRadixSelectGlobal(struct Point *data, struct Point *data_copy, int n, int *partition, int dir)
{
    cuRadixSelect(data, data_copy, n, partition, dir);
}

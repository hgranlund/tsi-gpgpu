#include "multiple-radix-select.cuh"
#include "common.cuh"
#include <stdio.h>

#include <helper_cuda.h>

#define checkCudaErrors(val)           check ( (val), #val, __FILE__, __LINE__ )

# define debug 0
#define FILE (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define debugf(fmt, ...) if(debug)printf("%s:%d: " fmt, FILE, __LINE__, __VA_ARGS__);


// __device__
// void printIntArray__(int *l, int n, char *s)
// {
// #if __CUDA_ARCH__ >= 200
//     if (debug && threadIdx.x == 0 && blockIdx.x == 1)
//     {
//         // printf("%s: ", s);
//         // printf("[%d", l[0] );
//         for (int i = 1; i < n; ++i)
//         {
//             // printf(", %d", l[i] );
//         }
//         // printf("]\n");
//     }
// #endif
// }

// __device__
// void d_print_points____(struct PointS *l, int n, char *s)
// {
//     __syncthreads();
//     if (threadIdx.x == 0 && blockIdx.x == 1)
//     {
//         int i;
//         // printf("%s: ", s);
//         // printf("[%3.1f", l[0].p[0] );
//         for (i = 1; i < n; ++i)
//         {
//             // printf(", %3.1f", l[i].p[0] );
//         }
//         // printf("]\n");
//     }
//     __syncthreads();
// }


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

__device__ int cuSumReduce(int *list, int n)
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
    __syncthreads();
    return list[0];
}

__device__ int cuPartitionSwap(struct PointS *data, struct PointS *swap, unsigned int n, int *partition, int *zero_count, int *one_count, struct PointS median, int dir)
{
    unsigned int
    tid = threadIdx.x,
    big,
    is_bigger,
    less;
    zero_count++;
    one_count++;
    zero_count[threadIdx.x] = 0;
    one_count[threadIdx.x] = 0;

    while (tid < n)
    {
        swap[tid] = data[tid];
        is_bigger = partition[tid] = (bool)(data[tid].p[dir] > median.p[dir]);
        one_count[threadIdx.x] += is_bigger;
        zero_count[threadIdx.x] += !is_bigger;
        tid += blockDim.x;
    }
    __syncthreads();
    cuAccumulateIndex(zero_count, blockDim.x);
    cuAccumulateIndex(one_count, blockDim.x);
    tid = threadIdx.x;
    one_count--;
    zero_count--;
    less = zero_count[threadIdx.x];
    big = one_count[threadIdx.x];
    __syncthreads();
    while (tid < n)
    {
        if (!partition[tid])
        {
            data[less] = swap[tid];
            less++;
        }
        else
        {
            data[n - big - 1] = swap[tid];
            big++;
        }
        tid += blockDim.x;
    }
    return zero_count[blockDim.x];
}

__device__ unsigned int cuPartition(struct PointS *data, unsigned int n, int *partition, int *zero_count, int last, unsigned int bit, int dir)
{
    unsigned int
    tid = threadIdx.x,
    is_one,
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

__device__ void cuRadixSelect(struct PointS *data, struct PointS *data_copy, int n, int *partition, int dir)
{
    __shared__ int one_count[1025];
    __shared__ int zeros_count[1025];
    __shared__ struct PointS median;
    __shared__ int no_of_median;

    int l = 0,
        u = n,
        m_u = ceil((float)n / 2),
        bit = 0,
        last = 2,
        tid = threadIdx.x;

    struct PointS local_median;

    if (tid == 0)
    {
        no_of_median = 0;
    }

    while (tid < n)
    {
        partition[tid] = last;
        tid += blockDim.x;
    }

    tid = threadIdx.x;
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

    __syncthreads();
    while (tid < n)
    {
        if (partition[tid] == last)
        {
            local_median = median = data[tid];
            unsigned int local_no_of_median = atomicAdd((unsigned int *)&no_of_median, 1);
            data[tid] = data[local_no_of_median];
            data[local_no_of_median] = median; //can be removed

        }
        tid += blockDim.x;
    }
    __syncthreads();

    int midpoint = cuPartitionSwap(data + no_of_median, data_copy, n - no_of_median, partition, one_count, zeros_count, median, dir);
    midpoint = midpoint + no_of_median - 1;
    __syncthreads();
    tid = threadIdx.x;

    while (tid < no_of_median)
    {
        data[tid] = data[midpoint - tid], data[midpoint - tid] = median;
        tid += blockDim.x;
    }
}

__global__
void cuBalanceBranch(struct PointS *points, struct PointS *swap, int *partition, int *steps, int p, int dir)
{

    int bid = blockIdx.x,
        blockoffset,
        n;
    while (bid < p)
    {
        blockoffset = steps[bid * 2];
        n = steps[bid * 2 + 1] - blockoffset;
        cuRadixSelect(points + blockoffset, swap + blockoffset, n, partition + blockoffset, dir);
        bid += gridDim.x;
    }
}

//For testing - One cannot import a __device__ kernel
__global__ void cuRadixSelectGlobal(struct PointS *data, struct PointS *data_copy, int n, int *partition, int dir)
{
    cuRadixSelect(data, data_copy, n, partition, dir);
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


void  multiRadixSelectAndPartition(struct PointS *d_data, struct PointS *d_data_copy, int *d_partition, int *d_steps, int n, int p,  int dir)
{
    int numBlocks, numThreads;
    getThreadAndBlockCountMulRadix(n, p, numBlocks, numThreads);
    cuBalanceBranch <<< numBlocks, numThreads>>>(d_data, d_data_copy, d_partition, d_steps, p, dir);
}


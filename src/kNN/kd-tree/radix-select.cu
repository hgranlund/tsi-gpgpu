#include "sum-reduction.cuh"
#include "radix-select.cuh"
#include <stdio.h>

#include <helper_cuda.h>

#define checkCudaErrors(val)           check ( (val), #val, __FILE__, __LINE__ )

# define debug 0
#define FILE (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define debugf(fmt, ...) if(debug)printf("%s:%d: " fmt, FILE, __LINE__, __VA_ARGS__);

//TODO:
// refactor
// Levere alternere mellom swap og points slik at man slipper å skrive til å fra swap.
// Ikke forandre plassering kun left and rigth????


__device__
void d_printIntArray___(int *l, int n, char *s)
{
#if __CUDA_ARCH__ >= 200
    if (debug && threadIdx.x == 0 )
    {
        int i;
        printf("%s: ", s);
        printf("[%d", l[0] );
        for (i = 1; i < n; ++i)
        {
            printf(", %d", l[i] );
        }
        printf("]\n");
        __syncthreads();
    }
#endif
}
__device__
void d_print_points___(struct PointS *l, int n, char *s)
{
    if (debug && threadIdx.x == 0)
    {
        int i;
        printf("%s: ", s);
        printf("[%3.1f", l[0].p[0] );
        for (i = 1; i < n; ++i)
        {
            printf(", %3.1f", l[i].p[0] );
        }
        printf("]\n");
    }
    __syncthreads();
}

void h_printIntArray___(int *l, int n, char *s)
{
    if (debug)
    {
        int i;
        printf("%s: ", s);
        printf("[%d", l[0]);
        for (i = 1; i < n; ++i)
        {
            printf(", %d", l[i] );
        }
        printf("]\n");
    }
}


int nextPowerOf2(int x)
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

__device__  void cuAccumulateIndex_(int *list, int n)
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


__global__ void cuPartitionSwap(struct PointS *points, struct PointS *swap, int n, int *partition, int last, int dir)
{
    __shared__ int ones[1025];
    __shared__ int zeros[1025];
    __shared__ int medians[1025];
    __shared__ struct PointS median;

    int
    tid = threadIdx.x,
    big,
    less,
    mid,
    point_difference,
    *zero_count = ones,
     *one_count = zeros,
      *median_count = medians;

    zero_count++;
    one_count++;
    median_count++;
    zero_count[threadIdx.x] = 0;
    one_count[threadIdx.x] = 0;
    median_count[threadIdx.x] = 0;

    while (tid < n)
    {
        if (partition[tid] == last)
        {
            median = points[tid];
        }
        tid += blockDim.x;
    }

    tid = threadIdx.x;
    __syncthreads();
    while (tid < n)
    {
        swap[tid] = points[tid];
        point_difference = partition[tid] = (points[tid].p[dir] - median.p[dir]);
        if (point_difference < 0)
        {
            zero_count[threadIdx.x]++;
        }
        else if (point_difference > 0)
        {
            one_count[threadIdx.x]++;
        }
        else
        {
            median_count[threadIdx.x]++;
        }
        tid += blockDim.x;
    }

    __syncthreads();
    cuAccumulateIndex_(zero_count, blockDim.x);
    cuAccumulateIndex_(one_count, blockDim.x);
    cuAccumulateIndex_(median_count, blockDim.x);
    __syncthreads();

    tid = threadIdx.x;
    one_count--;
    zero_count--;
    median_count--;
    mid = zero_count[blockDim.x] +  median_count[threadIdx.x];
    less = zero_count[threadIdx.x];
    big = one_count[threadIdx.x];
    while (tid < n)
    {
        if (partition[tid] < 0)
        {
            points[less] = swap[tid];
            less++;
        }
        else if (partition[tid] > 0)
        {
            points[n - big - 1] = swap[tid];
            big++;
        }
        else
        {
            points[mid] = swap[tid];
            mid++;
        }
        tid += blockDim.x;
    }
    __syncthreads();
}

__global__ void cuPartitionStep(struct PointS *data, unsigned int n, int *partition, int *zeros_count_block, int last, unsigned int bit, int dir)
{
    __shared__ int zero_count[1024];

    int
    tid = threadIdx.x,
    is_one,
    block_offset,
    rest = n % gridDim.x,
    block_step,
    radix = (1 << 31 - bit);
    zero_count[threadIdx.x] = 0;
    block_step = n / gridDim.x;
    n = block_step;
    block_offset = block_step * blockIdx.x;
    if (rest >= gridDim.x - blockIdx.x)
    {
        block_offset += rest - (gridDim.x - blockIdx.x);
        n++;
    }
    data += block_offset;
    partition += block_offset;
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
    cuSumReduce_(zero_count, blockDim.x);
    if (threadIdx.x == 0)
    {
        zeros_count_block[blockIdx.x] = zero_count[0];
    }
}

void getThreadAndBlockCountPartition(int n, int &blocks, int &threads)
{
    threads = min(nextPowerOf2(n), THREADS_PER_BLOCK_RADIX);
    blocks = n / threads / 2;
    blocks = max(1, nextPowerOf2(blocks));
    blocks = min(MAX_BLOCK_DIM_SIZE_RADIX, blocks);
}

__global__ void fillArray(int *array, int value, int n)
{
    int tid = threadIdx.x;
    while (tid < n)
    {
        array[tid] = value;
        tid += blockDim.x;
    }
}

void radixSelectAndPartition(struct PointS *points, struct PointS *swap, int *partition, int n, int dir)
{
    int numBlocks,
        numThreads,
        cut,
        l = 0,
        u = n,
        m_u = ceil((float)n / 2),
        bit = 0,
        last = 2,
        *h_zeros_count_block,
        *d_zeros_count_block;

    numThreads = min(n, 1024);
    fillArray <<< 1, numThreads>>>(partition, 2, n);
    getThreadAndBlockCountPartition(n, numBlocks, numThreads);
    debugf("blocks = %d, threads = %d, n = %d\n", numBlocks, numThreads, n);
    h_zeros_count_block = (int *) malloc(numBlocks * sizeof(int));
    checkCudaErrors(
        cudaMalloc((void **) &d_zeros_count_block, numBlocks * sizeof(int)));

    do
    {
        cuPartitionStep <<< numBlocks, numThreads>>>(points, n, partition, d_zeros_count_block, last, bit++, dir);

        sum_reduce(d_zeros_count_block, numBlocks);
        cudaMemcpy(h_zeros_count_block, d_zeros_count_block, sizeof(int), cudaMemcpyDeviceToHost);

        cut = h_zeros_count_block[0];

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

    cuPartitionSwap <<< 1, min(nextPowerOf2(n), MAX_BLOCK_DIM_SIZE_RADIX) >>> (points, swap, n, partition, last, dir);

    checkCudaErrors(
        cudaFree(d_zeros_count_block));
}





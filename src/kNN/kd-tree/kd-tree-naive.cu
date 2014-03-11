#include <kd-tree-naive.cuh>
#include "common.cuh"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <point.h>

#include <helper_cuda.h>

#define THREADS_PER_BLOCK 1024U
#define MAX_BLOCK_DIM_SIZE 65535U
#define MAX_SHARED_MEM 49152U
// #define THREADS_PER_BLOCK 4U
// #define MAX_BLOCK_DIM_SIZE 8U


#define debug 0
#include "common-debug.cuh"

__global__ void cuRadixSelectGlobal(Point *data, Point *data_copy, unsigned int m, unsigned int n, int *partition, int dir)
{
  cuRadixSelect(data, data_copy, m, n, partition, dir);
}

__device__ void cuPointSwap(Point *p, int a, int b){
    Point temp = p[a];
    p[a]=p[b], p[b]=temp;
}

__device__ void cuPartitionSwap(Point *data, Point *swap, unsigned int n, int *partition, int *zero_count, int *one_count, Point median, int dir)
{
    unsigned int
    tid = threadIdx.x,
    is_bigger,
    big,
    less;

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
            is_one = partition[tid]= (bool)((*(int*)&(data[tid].p[dir]))&radix);
            zero_count[threadIdx.x] += !is_one;
        }else{
            partition[tid] = 2;
        }
        tid+=blockDim.x;
    }
    return cuSumReduce(zero_count, blockDim.x);
}

__device__ void cuRadixSelect(Point *data, Point *data_copy, unsigned int m, unsigned int n, int *partition, int dir)
{
    __shared__ int one_count[1025];
    __shared__ int zeros_count[1025];
    __shared__ Point median;


    int l=0,
    u = n,
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
        if ((l+cut) <= m)
        {
            l +=cut;
            last = 1;
        }
        else
        {
            last = 0;
            u -=u-cut-l;
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
    median = data[m];
    data[m]=data[0], data[0] = median;
}

__global__
void cuBalanceBranchLeafs(Point* points, int n, int dir)
{
    int
    step = n/gridDim.x,
    blockOffset = step*blockIdx.x,
    tid = threadIdx.x;
    step=step/2;
    Point point1;
    Point point2;
    points += blockOffset;
    while(tid < step){
        point1 = points[tid*2];
        point2 = points[tid*2+1];
        if (point1.p[dir]>point2.p[dir])
        {
            points[tid*2] = point2;
            points[tid*2+1] = point1;
        }
        tid += blockDim.x;
    }
}

template <int maxStep> __global__
void cuQuickSelectShared(Point* points, int n, int p, int dir){
    __shared__ Point ss_points[maxStep*128];
    Point *s_points = ss_points;
    int pos, i, left, right,
    listInBlock = p/gridDim.x,
    tid = threadIdx.x,
    m=n/2;
    points += listInBlock * blockIdx.x * n;
    points += n * tid;
    s_points += (tid * maxStep);
    float pivot;
    while( tid < listInBlock)
    {
        for (i = 0; i < n; ++i)
        {
            s_points[i]=points[i];
        }
        left = 0;
        right = n - 1;
        while (left < right)
        {
            pivot = s_points[m].p[dir];
            cuPointSwap(s_points, m, right);
            for (i = pos = left; i < right; i++)
            {
                if (s_points[i].p[dir] < pivot)
                {
                    cuPointSwap(s_points, i, pos);
                    pos++;
                }
            }
            cuPointSwap(s_points, right, pos);
            if (pos == m) break;
            if (pos < m) left = pos + 1;
            else right = pos - 1;
        }
        for (i = 0; i <n; ++i)
        {
            points[i]=s_points[i];
        }
        tid += blockDim.x;
        points += n * blockDim.x;
    }
}

__global__
void cuQuickSelectGlobal(Point* points, int n, int p, int dir){
    int pos, i, left, right,

    listInBlock = p/gridDim.x,
    tid = threadIdx.x,
    m=n/2;
    points += listInBlock * blockIdx.x * n;
    points += n * tid;
    float pivot;
    while( tid < listInBlock)
    {
        left = 0;
        right = n - 1;
        while (left < right)
        {
            pivot = points[m].p[dir];
            cuPointSwap(points, m, right);
            for (i = pos = left; i < right; i++)
            {
                if (points[i].p[dir] < pivot)
                {
                    cuPointSwap(points, i, pos);
                    pos++;
                }
            }
            cuPointSwap(points, right, pos);
            if (pos == m) break;
            if (pos < m) left = pos + 1;
            else right = pos - 1;
        }
        tid += blockDim.x;
        points += n * blockDim.x;
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
void getThreadAndBlockCount(int n, int p, int &blocks, int &threads)
{
    n = n/p;
    n = prevPowTwo(n/2);
    blocks = min(MAX_BLOCK_DIM_SIZE, p);
    blocks = max(1, blocks);
    threads = min(THREADS_PER_BLOCK, n);
    threads = max(1, threads);
}
void getThreadAndBlockCountForQuickSelect(int n, int p, int &blocks, int &threads)
{
    int step = n/p,
    numberOfLists= p;
    threads = 128;
    blocks = numberOfLists/threads;
    blocks = min(MAX_BLOCK_DIM_SIZE, blocks);
    blocks = max(1, blocks);
}

void build_kd_tree(Point *h_points, int n)
{


    Point *d_points, *d_swap;
    int p, i, numBlocks, numThreads, step;
    int *d_partition;

    checkCudaErrors(
        cudaMalloc(&d_partition, n*sizeof(int)));

    checkCudaErrors(
        cudaMalloc(&d_points, n*sizeof(Point)));

    checkCudaErrors(
        cudaMalloc(&d_swap, n*sizeof(Point)));

    checkCudaErrors(
        cudaMemcpy(d_points, h_points, n*sizeof(Point), cudaMemcpyHostToDevice));

    p = 1;
    step = n/p;
    i = 0;
    while(step > 256)
    {
        getThreadAndBlockCount(n, p, numBlocks, numThreads);
        debugf("n = %d, p = %d, numblosck = %d, numThread =%d\n", n/p, p, numBlocks, numThreads );
        cuBalanceBranch<<<numBlocks,numThreads>>>(d_points, d_swap, d_partition, n/p, p, i%3);
        p <<=1;
        step=n/p;
        i++;
    }
    while(step > 2)
    {
        getThreadAndBlockCountForQuickSelect(n, p, numBlocks, numThreads);
        debugf("n = %d, p = %d, numblosck = %d, numThread =%d\n", n/p, p, numBlocks, numThreads );
        // printf("n = %d, p = %d, numblosck = %d, numThread =%d\n", n/p, p, numBlocks, numThreads );
        if (step > 16)
        {
            cuQuickSelectGlobal<<<numBlocks,numThreads>>>(d_points, n/p, p, i%3);
        }
        else if (step > 8)
        {
            if (16* sizeof(Point) * numThreads < MAX_SHARED_MEM)
            {
                cuQuickSelectShared<16><<<numBlocks,numThreads>>>(d_points, n/p, p, i%3);
            }
            else
            {
                cuQuickSelectGlobal<<<numBlocks,numThreads>>>(d_points, n/p, p, i%3);
            }
        }
        else if (step > 4)
        {
            if (8* sizeof(Point) * numThreads < MAX_SHARED_MEM)
            {
                cuQuickSelectShared<8><<<numBlocks,numThreads>>>(d_points, n/p, p, i%3);
            }
            else
            {
                cuQuickSelectGlobal<<<numBlocks,numThreads>>>(d_points, n/p, p, i%3);
            }
        }
        else
        {
            if (4* sizeof(Point) * numThreads < MAX_SHARED_MEM)
            {
                cuQuickSelectShared<4><<<numBlocks,numThreads>>>(d_points, n/p, p, i%3);
            }
            else
            {
                cuQuickSelectGlobal<<<numBlocks,numThreads>>>(d_points, n/p, p, i%3);
            }
        }
        p <<=1;
        step=n/p;
        i++;
    }

    numThreads = min(n, THREADS_PER_BLOCK/2);
    numBlocks = n/numThreads;
    numBlocks = min(numBlocks, 65536);
    debugf("n = %d, p = %d, numblosck = %d, numThread =%d\n", n/p, p, numBlocks, numThreads );
    cuBalanceBranchLeafs<<<numBlocks, numThreads>>>(d_points, n, i%3);

    checkCudaErrors(
        cudaMemcpy(h_points, d_points, n*sizeof(Point), cudaMemcpyDeviceToHost));


    checkCudaErrors(cudaFree(d_points));
    checkCudaErrors(cudaFree(d_swap));
    checkCudaErrors(cudaFree(d_partition));
}



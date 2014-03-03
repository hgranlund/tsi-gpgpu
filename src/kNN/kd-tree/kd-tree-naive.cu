#include "kd-tree-naive.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <helper_cuda.h>

#define checkCudaErrors(val)           check ( (val), #val, __FILE__, __LINE__ )
#define THREADS_PER_BLOCK 512U
#define MAX_BLOCK_DIM_SIZE 65535U
// #define THREADS_PER_BLOCK 4U
// #define MAX_BLOCK_DIM_SIZE 8U

#include <string.h>
#define debug 0
#define FILE (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define debugf(fmt, ...) if(debug){printf("%s:%d: " fmt, FILE, __LINE__, __VA_ARGS__);}


__constant__  int pitch;



int h_index(int i, int j, int n)
{
    return i + j * n;
}

void h_swap(float *points, int a, int b, int n)
{
    float tx = points[h_index(a, 0, n)],
    ty = points[h_index(a, 1, n)],
    tz = points[h_index(a, 2, n)];
    points[h_index(a, 0, n)] = points[h_index(b, 0, n)], points[h_index(b, 0, n)] = tx;
    points[h_index(a, 1, n)] = points[h_index(b, 1, n)], points[h_index(b, 1, n)] = ty;
    points[h_index(a, 2, n)] = points[h_index(b, 2, n)], points[h_index(b, 2, n)] = tz;
}

int midpoint(int lower, int upper)
{
    return (int) floor((upper - lower) / 2) + lower;
}

__device__ __host__
unsigned int nextPowerOf2(unsigned int x)
{
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return ++x;
}


__device__ int index(int i, int j, int offset)
{
    return i + (j * pitch) + offset;
}

__device__
bool isPowTwo(unsigned int x)
{
  return ((x&(x-1))==0);
}

__device__
unsigned int prevPowerOf2(unsigned int n){
    if (isPowTwo(n))
    {
        return n;
    }
    n = nextPowerOf2(n);
    return n >>=1;

}


void h_print_matrix(float* points, int n){
    if (debug)
    {
        printf("#################\n");
        for (int i = 0; i < n; ++i)
        {
            printf("i = %2d:   ", i );
            for (int j = 0; j < 3; ++j)
            {
                printf(  " %5.0f ", points[h_index(i,j,n)] );
            }
            printf("\n");
        }
    }
}

__device__
void printMatrix(float* points, int n, int offset){
#if __CUDA_ARCH__>=200
    if (debug)
    {
        __syncthreads();
        if (threadIdx.x ==0 && blockIdx.x ==0)
        {
            printf("####################\n");
            printf("block = %d, blockOffset = %d\n", blockIdx.x, offset );
            for (int i = 0; i < n; ++i)
            {
                printf("i = %3d", i );
                for (int j = 0; j < 3; ++j)
                {
                    printf(  " %3.1f ", points[index(i,j, offset)] );
                }
                printf("\n",1 );
            }
            printf("\n####################\n");
        }
        __syncthreads();
    }
#endif
}

__device__
void swap(float *points, int a, int b, int offset)
{
    float tx = points[index(a, 0, offset)],
    ty = points[index(a, 1, offset)],
    tz = points[index(a, 2, offset)];
    points[index(a, 0, offset)] = points[index(b, 0, offset)], points[index(b, 0, offset)] = tx;
    points[index(a, 1, offset)] = points[index(b, 1, offset)], points[index(b, 1, offset)] = ty;
    points[index(a, 2, offset)] = points[index(b, 2, offset)], points[index(b, 2, offset)] = tz;
}


__device__
void cuCompare(float* points, int a, int b, int ddd, int n, int offset, int dir){
    if (b < n)
    {
        if ((points[index(a,dir,offset)] >= points[index(b,dir,offset)]) == ddd)
        {
            swap(points, a, b, offset);
        }
    }
}

__device__
void cuBitonicSortOneBlock(float *points, unsigned int n, unsigned int offset, int dir)
{
    unsigned int size, stride, ddd, pos, tid;
    unsigned int i, m;

    for (size = 2; size <= nextPowerOf2(n); size <<= 1)
    {
        tid=threadIdx.x;
        ddd = 1 ^ ((tid & (size / 2)) != 0);
        for (stride = size / 2; stride > 0; stride >>= 1)
        {
            tid = threadIdx.x;
            __syncthreads();
            while(tid < n/2)
            {
                ddd = 1 ^ ((tid & (size / 2)) != 0);
                pos = 2 * tid - (tid & (stride - 1));
                cuCompare(points, pos, pos+stride, ddd, n, offset, dir);
                tid+=blockDim.x;
            }
        }
    }
}



// A shorted bitonic sort is used to find and place the median.
__global__
void cuBalanceBranch(float* points, int n, unsigned int p, int dir){

    unsigned int prev_pow_two, i, step, tid, blockoffset, bid;
    tid = threadIdx.x;
    bid = blockIdx.x;
    while(bid < p){
        step = (int) (n / p);
        blockoffset = (step) * bid;
        n=step;
        prev_pow_two = prevPowerOf2(n);
        printMatrix(points, prev_pow_two, blockoffset);
        cuBitonicSortOneBlock(points, n, blockoffset, dir);
        // if (!isPowTwo(n))
        // {
        // __syncthreads();
        //     printMatrix(points, n, blockoffset);
        //     while(tid < (n-prev_pow_two))
        //     {
        //         swap(points, prev_pow_two-tid, prev_pow_two+tid, blockoffset);
        //         tid+=blockDim.x;
        //     }
        //     __syncthreads();
        //     cuBitonicSortOneBlock(points, n, blockoffset, dir);
        //     __syncthreads();
        //     while(tid < (n-prev_pow_two))
        //     {
        //         swap(points, prev_pow_two-tid, prev_pow_two+tid, blockoffset);
        //         tid+=blockDim.x;
        //     }
        //     __syncthreads();
        //     cuBitonicSortOneBlock(points, prev_pow_two, (n-prev_pow_two)+blockoffset, dir);
        //     printMatrix(points, n, blockoffset);
        // }
        bid += gridDim.x;
    }

}

void getThreadAndBlockCount(int n, int p, int &blocks, int &threads)
{
    blocks = min(MAX_BLOCK_DIM_SIZE, p);
    blocks = max(1, blocks);
    threads = min(THREADS_PER_BLOCK, (n/p)/2);
}

void build_kd_tree(float *h_points, int n)
{

    h_print_matrix(h_points, n);

    float* d_points;
    int p, h, i, numBlocks, numThreads, d_pitch, dim = 3;
    size_t d_pitch_in_bytes, h_pitch_in_bytes = n*sizeof(float);


    checkCudaErrors(
        cudaMallocPitch(&d_points, &d_pitch_in_bytes, n*sizeof(float), dim));
    d_pitch    = d_pitch_in_bytes/sizeof(float);

    checkCudaErrors(cudaMemcpyToSymbol(pitch, &d_pitch, sizeof(int)));

    checkCudaErrors(
        cudaMemcpy2D(d_points, d_pitch_in_bytes, h_points, h_pitch_in_bytes, n*sizeof(float), dim, cudaMemcpyHostToDevice));
    h = ceil(log2((float)n + 1) - 1);
    for (i = 0; i < h; i++)
    {
        p = pow(2, i);
        getThreadAndBlockCount(n, p, numBlocks, numThreads);
        debugf("n = %d, p = %d, numblosck = %d, numThread =%d\n", n, p, numBlocks, numThreads );
        cuBalanceBranch<<<numBlocks,numThreads>>>(d_points, n, p, i%3);
    }

    checkCudaErrors(
        cudaMemcpy2D(h_points, h_pitch_in_bytes, d_points, d_pitch_in_bytes, n*sizeof(float), dim, cudaMemcpyDeviceToHost));
    h_print_matrix(h_points, n);
    checkCudaErrors(cudaFree(d_points));
}



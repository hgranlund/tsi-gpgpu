#include "kd-tree-naive.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <helper_cuda.h>

#define checkCudaErrors(val)           check ( (val), #val, __FILE__, __LINE__ )
// #define THREADS_PER_BLOCK 512U
// #define MAX_BLOCK_DIM_SIZE 65535U
#define THREADS_PER_BLOCK 4U
#define MAX_BLOCK_DIM_SIZE 8U


__constant__  size_t pitch;



int index(int i, int j, int n)
{
    return i + j * n;
}

void swap(float *points, int a, int b, int n)
{
    float tx = points[index(a, 0, n)],
    ty = points[index(a, 1, n)],
    tz = points[index(a, 2, n)];
    points[index(a, 0, n)] = points[index(b, 0, n)], points[index(b, 0, n)] = tx;
    points[index(a, 1, n)] = points[index(b, 1, n)], points[index(b, 1, n)] = ty;
    points[index(a, 2, n)] = points[index(b, 2, n)], points[index(b, 2, n)] = tz;
}

int midpoint(int lower, int upper)
{
    return (int) floor((upper - lower) / 2) + lower;
}

float quick_select(int k, float *points, int lower, int upper, int dim, int n)
{
    int pos, i,
    left = lower,
    right = upper - 1;

    float pivot;

    while (left < right)
    {
        pivot = points[index(k, dim, n)];
        swap(points, k, right, n);
        for (i = pos = left; i < right; i++)
        {
            if (points[index(i, dim, n)] < pivot)
            {
                swap(points, i, pos, n);
                pos++;
            }
        }
        swap(points, right, pos, n);
        if (pos == k) break;
        if (pos < k) left = pos + 1;
        else right = pos - 1;
    }
    return points[index(k, dim, n)];
}

void center_median(float *points, int lower, int upper, int dim, int n)
{
    int i, r = midpoint(lower, upper);

    float median = quick_select(r, points, lower, upper, dim, n);

    for (i = lower; i < upper; ++i)
    {
        if (points[index(i, dim, n)] == median)
        {
            swap(points, i, r, n);
            return;
        }
    }
}

void balance_branch(float *points, int lower, int upper, int dim, int n)
{
    if (lower >= upper) return;

    int i, r = midpoint(lower, upper);

    center_median(points, lower, upper, dim, n);

    upper--;

    for (i = lower; i < r; ++i)
    {
        if (points[index(i, dim, n)] > points[index(r, dim, n)])
        {
            while (points[index(upper, dim, n)] > points[index(r, dim, n)])
            {
                upper--;
            }
            swap(points, i, upper, n);
        }
    }
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


__device__
int index(int i, int j)
{
    return i + j * pitch;
}


__device__
void printMatrix(float* points, int n){
#if __CUDA_ARCH__>=200

    if (threadIdx.x ==0)
    {
    /* code */
        for (int i = 0; i < n; ++i)
        {
            printf("i = %3d", i );
            for (int j = 0; j < 3; ++j)
            {
                printf(  " %3.1f ", points[index(i,j)] );
            }
            printf("\n" );
        }
    }
#endif
}

__device__
void swap(float *points, int a, int b)
{
    float tx = points[index(a, 0)],
    ty = points[index(a, 1)],
    tz = points[index(a, 2)];
    points[index(a, 0)] = points[index(b, 0)], points[index(b, 0)] = tx;
    points[index(a, 1)] = points[index(b, 1)], points[index(b, 1)] = ty;
    points[index(a, 2)] = points[index(b, 2)], points[index(b, 2)] = tz;
}


__device__
void cuCompare(float* points, int a, int b, int ddd, int n, int dir){
    if (b < n)
    {
        #if __CUDA_ARCH__>=200
        // printf("block/Thread: %d/%d, swap:%1d,  %4d <--> %4d #### %3.1f  <--> %3.1f\n", blockIdx.x, threadIdx.x, (points[index(a,dir)] >=points[index(b,dir)]) == ddd, a, b, points[index(a, 0)], points[index(b, 0)]  );
        #endif
        if ((points[index(a,dir)] >=points[index(b,dir)]) == ddd)
        {
            swap(points, a, b);
        }
    }
}

__global__
void cuBalanceBranch(float* points, int n, int numBranch, int dir){

// int blockoffset = blockIdx.x * blockDim.x *2;
// dist+=blockoffset;
// ind+=blockoffset;


    unsigned int size, m, i, stride, ddd, pos,
    tid = threadIdx.x;

    for (size = 2; size <= nextPowerOf2(n); size <<= 1)
    {
        tid=threadIdx.x;
        ddd = 1 ^ ((tid & (size / 2)) != 0);

        if (size > n)
        {
            i = tid;
            m=size;
            m >>=1;
            while(i < n-m)
            {
                ddd = 1 ^ ((tid & (size / 2)) != 0);
                cuCompare(points, i, m-i-1, !ddd, n, dir);
                i+=blockDim.x;
            }
        }

        __syncthreads();
        for (stride = size / 2; stride > 0; stride >>= 1)
        {
            tid = threadIdx.x;
            __syncthreads();
            while(tid < n/2){
                ddd = 1 ^ ((tid & (size / 2)) != 0);
                pos = 2 * tid - (tid & (stride - 1));
                cuCompare(points, pos, pos+stride, ddd, n, dir);
                tid+=blockDim.x;
            }
        }
    }
}

void getThreadAndBlockCount(int n, int p, int &blocks, int &threads)
{
    blocks = min(MAX_BLOCK_DIM_SIZE, p);
    threads = min(THREADS_PER_BLOCK, n/2);
}

void build_kd_tree(float *h_points, int n)
{
    float* d_points;
    int step, h,p, numBlocks, numThreads,
    dim = 3;
    size_t d_pitch_in_bytes,d_pitch, h_pitch_in_bytes = n*sizeof(float);

    h = ceil(log2((float)n + 1) - 1);

    checkCudaErrors(
        cudaMallocPitch(&d_points, &d_pitch_in_bytes, n*sizeof(float), dim));
    d_pitch    = d_pitch_in_bytes/sizeof(float);
    checkCudaErrors(cudaMemcpyToSymbol(pitch, &d_pitch, sizeof(size_t)));


    checkCudaErrors(
        cudaMemcpy2D(d_points, d_pitch_in_bytes, h_points, h_pitch_in_bytes, n*sizeof(float), dim, cudaMemcpyHostToDevice));
    p=1;
    getThreadAndBlockCount(n, p, numBlocks, numThreads);
    cuBalanceBranch<<<numBlocks,numThreads>>>(d_points, n, 1, 0);

    checkCudaErrors(
        cudaMemcpy2D(h_points, h_pitch_in_bytes, d_points, d_pitch_in_bytes, n*sizeof(float), dim, cudaMemcpyDeviceToHost));

    // printf("#################\n");
    // for (int i = 0; i < n; ++i)
    // {
    //     printf("i = %2d:   ", i );
    //     for (int j = 0; j < 3; ++j)
    //     {
    //         printf(  " %5.0f ", h_points[index(i,j,n)] );
    //     }
    //     printf("\n" );
    // }
    // for (i = 0; i < h; ++i)
    // {
    //     step = (int) floor(n / pow(2, i)) + 1;
    //     for (j = 0; j < n; j+=step)
    //     {
    //         balance_branch(h_points, j, j+step-1, i%dim, n);
    //     }
    // }

    checkCudaErrors(cudaFree(d_points));
}



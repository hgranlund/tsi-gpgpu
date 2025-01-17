#include "reduction.cuh"
#include "cuda.h"
#include "stdio.h"



// #define SHARED_SIZE_LIMIT 8U
#define SHARED_SIZE_LIMIT 512U

__device__ unsigned int nextPow2(unsigned int x)
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}


// Can be optimized
__device__ int nearestPowerOf2 (int n)
{
    if (!n)
    {
        return n;  //(0 == 2^0)
    }
    int x = 1;
    while (x <= n)
    {
        x <<= 1;
    }
    return x;
}

void compare(Distance &distA,  Distance &distB, unsigned int dir)
{
    Distance f;
    if ((distA.value  >= distB.value) == dir)
    {
        f = distA;
        distA  = distB;
        distB = f;
    }
}

__device__ void cuCompare_r(Distance &distA,  Distance &distB, unsigned int dir)
{
    Distance f;
    if ((distA.value  >= distB.value) == dir)
    {
        f = distA;
        distA  = distB;
        distB = f;
    }
}



__global__ void min_reduction(Distance *dist, unsigned int n, unsigned int threadOffset)
{

    unsigned int  thread1, halfstruct Point, index1, index2, offset;
    unsigned int threadOffset1 = max(1, threadOffset);
    unsigned int elements_in_block = nextPow2(n);
    offset = elements_in_block - n;
    dist += blockIdx.x * n;
    while (elements_in_block > 1)
    {
        thread1 = threadIdx.x;
        halfstruct Node = (elements_in_block / 2);
        while (thread1 < halfstruct Point)
        {
            if (thread1 + halfstruct Node   < elements_in_block - offset)
            {
                index1 = thread1 * threadOffset1;
                index2 = index1  + halfstruct Node * threadOffset1;
                cuCompare_r(dist[index1], dist[index2], 1);
            }
            thread1 += blockDim.x;
        }
        __syncthreads();
        offset = 0;
        elements_in_block = halfstruct Point;
    }
}

void knn_min_reduce(Distance *d_dist, unsigned int n)
{
    unsigned int blockCount, threadCount, elements_in_block, elements_out_of_block, offset;
    blockCount = ceil((float)n / SHARED_SIZE_LIMIT );
    elements_in_block = n / blockCount;
    if (blockCount == 0)
    {
        elements_in_block = n;
        blockCount = 1;
    }
    threadCount = elements_in_block / 2;
    threadCount = min(SHARED_SIZE_LIMIT, threadCount);
    elements_out_of_block = n - blockCount * elements_in_block;
    if (elements_out_of_block > 0 )
    {
        offset = n - (elements_out_of_block * 2);
        min_reduction <<< 1, elements_out_of_block>>>(d_dist + offset + offset, elements_out_of_block * 2, 0);
    }
    min_reduction <<< blockCount, threadCount>>>(d_dist, elements_in_block, 0);
    min_reduction <<< 1, blockCount>>>(d_dist, blockCount, elements_in_block);
}



void min_reduce(Distance *h_dist, int *h_ind, unsigned int n)
{

    Distance *d_dist;
    unsigned int blockCount, threadCount, elements_in_block;
    blockCount = ceil((float)n / SHARED_SIZE_LIMIT );
    elements_in_block = n / blockCount;
    if (blockCount == 0)
    {
        elements_in_block = n;
        blockCount = 1;
    }
    threadCount = elements_in_block  / 2;
    threadCount = min(SHARED_SIZE_LIMIT, threadCount);

    for (int i = n - 1; i >= elements_in_block * blockCount; --i)
    {
        compare(h_dist[0], h_dist[i], 1);

    }
    cudaMalloc( (void **) &d_dist, n * sizeof(Distance));

    cudaMemcpy(d_dist, h_dist, n * sizeof(Distance), cudaMemcpyHostToDevice);
    min_reduction <<< blockCount, threadCount>>>(d_dist, elements_in_block, 0);
    min_reduction <<< 1, blockCount>>>(d_dist, blockCount, elements_in_block);


    cudaMemcpy(h_dist, d_dist, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_dist);
}

#include "kd-tree-naive.cuh"
#include "multiple-radix-select.cuh"
#include "quick-select.cuh"
#include "radix-select.cuh"

#include "stdio.h"
#include "point.h"

#include "helper_cuda.h"

#define debug 0
#include "common-debug.cuh"


__global__
void cuBalanceBranchLeafs(Point *points, int n, int dir)
{
    int
    step = n / gridDim.x,
    blockOffset = step * blockIdx.x,
    tid = threadIdx.x;
    step = step >> 1;           // same as n / 2;
    Point point1;
    Point point2;
    points += blockOffset;
    while (tid < step)
    {
        point1 = points[tid * 2];
        point2 = points[tid * 2 + 1];
        if (point1.p[dir] > point2.p[dir])
        {
            points[tid * 2] = point2;
            points[tid * 2 + 1] = point1;
        }
        tid += blockDim.x;
    }
}

void nextStep(int *steps_new, int *steps_old, int n)
{
    int midpoint;
    for (int i = 0; i < n / 2; ++i)
    {
        midpoint = steps_old[i * 2 + 1] / steps_old[i * 2];
        steps_new[i * 4] = steps_old[i * 2];
        steps_new[i * 4 + 1] = steps_old[midpoint];
        steps_new[i * 4 + 2] = steps_old[midpoint + 1];
        steps_new[i * 4 + 3] = steps_old[i * 2 + 1];
    }
}


void build_kd_tree(Point *h_points, int n)
{


    Point *d_points, *d_swap;
    int p, i, j, numBlocks, numThreads, step;
    int *d_partition , *d_steps, *h_steps_old, *h_steps_new;

    h_steps_new = (int *)malloc(n * sizeof(int));
    h_steps_old = (int *)malloc(n * sizeof(int));

    checkCudaErrors(
        cudaMalloc(&d_steps, n * sizeof(int)));

    checkCudaErrors(
        cudaMalloc(&d_partition, n * sizeof(int)));

    checkCudaErrors(
        cudaMalloc(&d_points, n * sizeof(Point)));

    checkCudaErrors(
        cudaMalloc(&d_swap, n * sizeof(Point)));

    checkCudaErrors(
        cudaMemcpy(d_points, h_points, n * sizeof(Point), cudaMemcpyHostToDevice));

    p = 1;
    step = n / p;
    i = 0;
    h_steps_new[0] = 0;
    h_steps_old[0] = 0;
    h_steps_old[1] = n;
    h_steps_new[1] = n;
    while (step >= 8388608 && p <= 2)
    {
        int nn, offset;
        for (j = 0; j < p; j ++)
        {
            offset = h_steps_new[j * 2];
            nn = h_steps_new[j * 2 + 1] - offset;
            radixSelectAndPartition(d_points + offset, d_swap + offset, d_partition + offset, nn, i % 3);
        }
        nextStep(h_steps_new, h_steps_old, p <<= 1);
        // p <<= 1;
        i++;
    }
    while (step > 256)
    {
        // nextStep(h_steps_new, h_steps_old, p <<= 1);
        // checkCudaErrors(
        // cudaMemcpy(d_steps, h_steps_new, p * sizeof(int), cudaMemcpyHostToDevice));
        multiRadixSelectAndPartition(d_points, d_swap, d_partition, step, p, i % 3);
        // p <<= 1;
        step = n / p;
        i++;
    }
    while (step > 2)
    {
        quickSelectAndPartition(d_points, step, p, i % 3);
        p <<= 1;
        step = n / p;
        i++;
    }

    numThreads = min(n, THREADS_PER_BLOCK / 2);
    numBlocks = n / numThreads;
    numBlocks = min(numBlocks, MAX_BLOCK_DIM_SIZE);
    debugf("n = %d, p = %d, numblosck = %d, numThread =%d\n", n / p, p, numBlocks, numThreads );
    cuBalanceBranchLeafs <<< numBlocks, numThreads>>>(d_points, n, i % 3);

    checkCudaErrors(
        cudaMemcpy(h_points, d_points, n * sizeof(Point), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_points));
    checkCudaErrors(cudaFree(d_swap));
    checkCudaErrors(cudaFree(d_partition));
}



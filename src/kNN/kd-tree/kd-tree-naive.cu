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
    int midpoint, from, to;
    for (int i = 0; i < n / 2; ++i)
    {
        from = steps_old[i * 2];
        to = steps_old[i * 2 + 1];
        midpoint = (to - from) / 2 + from;
        steps_new[i * 4] = from;
        steps_new[i * 4 + 1] = midpoint;
        steps_new[i * 4 + 2] = midpoint + 1;
        steps_new[i * 4 + 3] = to;
    }
}

void swap_pointer(int **a, int **b)
{
    int *swap;
    swap = *a, *a = *b, *b = swap;

}

void   singleRadixSelectAndPartition(Point *d_points, Point *d_swap, int *d_partition, int *h_steps, int p, int  dir)
{
    int nn, offset, j;
    for (j = 0; j < p; j ++)
    {
        offset = h_steps[j * 2];
        nn = h_steps[j * 2 + 1] - offset;
        if (nn > 1)
        {
            radixSelectAndPartition(d_points + offset, d_swap + offset, d_partition + offset, nn, dir);
        }
    }
}

void build_kd_tree(Point *h_points, int n)
{
    Point *d_points, *d_swap;
    int p, h, i, j, *d_partition,
        *d_steps, *h_steps_old, *h_steps_new;

    h_steps_new = (int *)malloc(n * 2 * sizeof(int));
    h_steps_old = (int *)malloc(n * 2 * sizeof(int));

    checkCudaErrors(
        cudaMalloc(&d_steps, n * 2 * sizeof(int)));

    checkCudaErrors(
        cudaMalloc(&d_partition, n * sizeof(int)));

    checkCudaErrors(
        cudaMalloc(&d_points, n * sizeof(Point)));

    checkCudaErrors(
        cudaMalloc(&d_swap, n * sizeof(Point)));

    checkCudaErrors(
        cudaMemcpy(d_points, h_points, n * sizeof(Point), cudaMemcpyHostToDevice));

    p = 1;
    i = 0;
    h = ceil(log2((float)n + 1));
    h_steps_new[0] = 0;
    h_steps_old[0] = 0;
    h_steps_old[1] = n;
    h_steps_new[1] = n;

    radixSelectAndPartition(d_points, d_swap, d_partition, n, i % 3);
    i++;
    while (i < h )
    {
        nextStep(h_steps_new, h_steps_old, p <<= 1);
        checkCudaErrors(
            cudaMemcpy(d_steps, h_steps_new, p * 2 * sizeof(int), cudaMemcpyHostToDevice));
        if (n / p >= 8388608)
        {
            singleRadixSelectAndPartition(d_points, d_swap, d_partition, h_steps_new, p, i % 3);
        }

        else if (n / p > 256)
        {
            multiRadixSelectAndPartition(d_points, d_swap, d_partition, d_steps, n, p, i % 3);
        }
        else
        {
            quickSelectAndPartition(d_points, d_steps, n, p, i % 3);
        }
        swap_pointer(&h_steps_new, &h_steps_old);
        i++;
    }

    checkCudaErrors(
        cudaMemcpy(h_points, d_points, n * sizeof(Point), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_points));
    checkCudaErrors(cudaFree(d_swap));
    checkCudaErrors(cudaFree(d_partition));
}



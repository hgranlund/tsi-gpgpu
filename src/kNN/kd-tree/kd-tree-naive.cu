#include "kd-tree-naive.cuh"
#include "multiple-radix-select.cuh"
#include "quick-select.cuh"
#include "radix-select.cuh"

#include "stdio.h"
#include "point.h"

#include "helper_cuda.h"

#define debug 0
#include "common-debug.cuh"


int store_locations(Point *tree, int lower, int upper, int n)
{
    int r;

    if (lower >= upper)
    {
        return -1;
    }

    r = (int) ((upper - lower) / 2) + lower;

    tree[r].left = store_locations(tree, lower, r, n);
    tree[r].right = store_locations(tree, r + 1, upper, n);

    return r;
}

__global__
void convertPoints( PointS *points_small, int n, Point *points)
{
    int
    block_stride = n / gridDim.x,
    block_offset = block_stride * blockIdx.x,
    tid = threadIdx.x,
    rest = n % gridDim.x;
    PointS point_s;
    if (rest >= gridDim.x - blockIdx.x)
    {
        block_offset += rest - (gridDim.x - blockIdx.x);
        block_stride++;
    }
    points += block_offset;
    while (tid < block_stride)
    {
        Point point;
        point_s = points_small[tid];
        point.p[0] = point_s.p[0];
        point.p[1] = point_s.p[1];
        point.p[2] = point_s.p[2];
        points[tid] = point;
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

void singleRadixSelectAndPartition(PointS *d_points, PointS *d_swap, int *d_partition, int *h_steps, int p, int  dir)
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

void build_kd_tree(PointS *h_points, int n, Point *h_points_out)
{
    PointS *d_points, *d_swap;
    Point *d_points_out;
    int p, h, i, *d_partition,
        *d_steps, *h_steps_old, *h_steps_new;

    h_steps_new = (int *)malloc(n * 2 * sizeof(int));
    h_steps_old = (int *)malloc(n * 2 * sizeof(int));

    checkCudaErrors(
        cudaMalloc(&d_steps, n * 2 * sizeof(int)));

    checkCudaErrors(
        cudaMalloc(&d_partition, n * sizeof(int)));

    checkCudaErrors(
        cudaMalloc(&d_points, n * sizeof(PointS)));

    checkCudaErrors(
        cudaMalloc(&d_swap, n * sizeof(PointS)));

    checkCudaErrors(
        cudaMemcpy(d_points, h_points, n * sizeof(PointS), cudaMemcpyHostToDevice));

    p = 1;
    i = 0;
    int step;
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
        step = h_steps_new[1] - h_steps_new[0];
        checkCudaErrors(
            cudaMemcpy(d_steps, h_steps_new, p * 2 * sizeof(int), cudaMemcpyHostToDevice));
        if (step >= 8388608)
        {
            singleRadixSelectAndPartition(d_points, d_swap, d_partition, h_steps_new, p, i % 3);
        }

        else if (step > 4000)
        {
            multiRadixSelectAndPartition(d_points, d_swap, d_partition, d_steps, step, p, i % 3);
        }
        else
        {
            quickSelectAndPartition(d_points, d_steps, step, p, i % 3);
        }
        swap_pointer(&h_steps_new, &h_steps_old);
        i++;
    }

    checkCudaErrors(cudaFree(d_swap));
    checkCudaErrors(cudaFree(d_partition));
    checkCudaErrors(
        cudaMalloc(&d_points_out, n * sizeof(Point)));

    convertPoints <<< max(1, n / 512), 512 >>> (d_points, n, d_points_out);
    checkCudaErrors(
        cudaMemcpy(h_points_out, d_points_out, n * sizeof(Point), cudaMemcpyDeviceToHost));

    store_locations(h_points_out, 0, n, n);

    checkCudaErrors(cudaFree(d_points));
    checkCudaErrors(cudaFree(d_steps));
}



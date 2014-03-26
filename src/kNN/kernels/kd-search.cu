#include <stdlib.h>
#include <math.h>

#include <helper_cuda.h>

#include "kd-search.cuh"

int store_locations(Point *tree, int lower, int upper, int n)
{
    int r;

    if (lower >= upper)
    {
        return -1;
    }

    r = (int) floor((upper - lower) / 2) + lower;

    tree[r].left = store_locations(tree, lower, r, n);
    tree[r].right = store_locations(tree, r + 1, upper, n);

    return r;
}

__device__
int nn(Point qp, Point *tree, int n, int k)
{
    return 1;
}

__global__
void dQueryAll(Point *query_points, Point *tree, int n_qp, int n_tree, int k, int *result)
{
    int tid = threadIdx.x,
        rest = n_qp % gridDim.x,
        block_step = n_qp / gridDim.x,
        block_offset = block_step * blockIdx.x;

    if (rest >= gridDim.x - blockIdx.x)
    {
        block_offset += rest - (gridDim.x - blockIdx.x);
        block_step++;
    }
    query_points += block_offset;
    // printf("blockIdx: %d, threadIdx: %d, gridDim: %d, blockDim: %d\n", blockIdx.x, threadIdx.x, gridDim.x, blockDim.x);
    while (tid < block_step)
    {
        result[tid] = nn(query_points[tid], tree, n_tree, k);
        tid += blockDim.x;
    }
}

void getThreadAndBlockCountForQueryAll(int n, int &blocks, int &threads)
{
    threads = 128;
    blocks = n / threads;
    blocks = min(MAX_BLOCK_DIM_SIZE, blocks);
    blocks = max(1, blocks);
}

void queryAll(Point *h_query_points, Point *h_tree, int n_qp, int n_tree, int k, int *h_result)
{
    int *d_result, numBlocks, numThreads;
    Point *d_tree, *d_query_points;


    checkCudaErrors(cudaMalloc(&d_result, n_qp * k  * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_query_points, n_qp * sizeof(Point)));
    checkCudaErrors(cudaMalloc(&d_tree, n_tree * sizeof(Point)));

    checkCudaErrors(cudaMemcpy(d_query_points, h_query_points, n_qp * sizeof(Point), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_tree, h_tree, n_tree * sizeof(Point), cudaMemcpyHostToDevice));

    getThreadAndBlockCountForQueryAll(n_qp, numBlocks, numThreads);
    dQueryAll <<< numBlocks, numThreads>>>(d_tree, d_tree, n_qp, n_tree, k, d_result);

    checkCudaErrors(cudaMemcpy(h_result, d_result, n_qp * k * sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_tree));
    checkCudaErrors(cudaFree(d_query_points));
    checkCudaErrors(cudaFree(d_result));
}

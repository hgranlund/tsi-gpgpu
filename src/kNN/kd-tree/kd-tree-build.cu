#include "kd-tree-build.cuh"
#include "multiple-radix-select.cuh"
#include "quick-select.cuh"
#include "radix-select.cuh"

#include "stdio.h"
#include "point.h"

#include "helper_cuda.h"

int nextPowerOf2_(int x)
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

void UpDim(int &dim)
{
    dim = (dim + 1) % 3;
}

void getThreadAndBlockCountForBuild(int n, int &blocks, int &threads)
{
    threads = min(nextPowerOf2_(n), 512);
    blocks = n / threads;
    blocks = max(1, blocks);
    blocks = min(MAX_BLOCK_DIM_SIZE, blocks);
    // printf("block = %d, threads = %d, n = %d\n", blocks, threads, n);
}

__device__ void cuCalculateBlockOffsetAndNoOfLists_(int n, int &n_per_block, int &block_offset)
{
    int rest = n % gridDim.x;

    n_per_block = n / gridDim.x;
    block_offset = n_per_block * blockIdx.x;

    if (rest >= gridDim.x - blockIdx.x)
    {
        block_offset += rest - (gridDim.x - blockIdx.x);
        n_per_block++;
    }
}

__device__ void cuPointSwapCondition(struct Point *p, int a, int b, int dim)
{
    struct Point temp_a = p[a], temp_b = p[b];
    if (temp_a.p[dim] > temp_b.p[dim] )
    {
        p[a] = temp_b, p[b] = temp_a;
    }
}

__global__ void balanceLeafs(struct Point *points, int *steps, int p, int dim)
{
    struct Point   *l_points;

    int list_in_block,
        block_offset,
        tid = threadIdx.x,
        step_num,
        n;

    cuCalculateBlockOffsetAndNoOfLists_(p, list_in_block, block_offset);

    steps += block_offset * 2;

    while ( tid < list_in_block)
    {
        step_num =  tid * 2;
        l_points = points + steps[step_num];
        n = steps[step_num + 1] - steps[step_num];
        if (n == 2)
        {
            cuPointSwapCondition(l_points, 0, 1, dim);
        }
        else if (n == 3)
        {
            cuPointSwapCondition(l_points, 0, 1, dim);
            cuPointSwapCondition(l_points, 1, 2, dim);
            cuPointSwapCondition(l_points, 0, 1, dim);
        }
        tid += blockDim.x;
    }
}

int store_locations(struct Node *tree, int lower, int upper, int n)
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

__device__ __host__
void pointConvert(struct Node &p1, struct Point &p2)
{
    p1.p[0] = p2.p[0], p1.p[1] = p2.p[1], p1.p[2] = p2.p[2];
#if KEEP_POINT_INDEX
    p1.index_orig = p2.index_orig;
#endif
}

__global__
void convertPoints(struct Point *points_small, int n, struct Node *points)
{
    int local_n,
        block_offset,
        tid = threadIdx.x;

    cuCalculateBlockOffsetAndNoOfLists_(n, local_n, block_offset);

    points += block_offset;
    points_small += block_offset;

    while (tid < local_n)
    {
        pointConvert(points[tid], points_small[tid]);
        tid += blockDim.x;
    }
}

void nextStep(int *steps_new, int *steps_old, int n)
{
    int i, midpoint, from, to;
    for (i = 0; i < n / 2; ++i)
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

void singleRadixSelectAndPartition(struct Point *d_points, struct Point *d_swap, int *d_partition, int *h_steps, int p, int  dir)
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

size_t getFreeBytesOnGpu_()
{
    size_t free_byte, total_byte ;
    cudaError_t cuda_status = cudaMemGetInfo( &free_byte, &total_byte ) ;
    return free_byte;
}

size_t getNeededBytesForBuildingKdTree(int n)
{
    int number_of_leafs = (n + 1) / 2;
    return (number_of_leafs * 2 * sizeof(int)) + (2 * n * sizeof(int)) + (2 * n * sizeof(Point));
}

void cuBuildKdTree(struct Point *h_points, int n, int dim, struct Node *tree)
{
    struct Point *d_points, *d_swap;
    struct Node *d_tree;
    int *d_partition,
        block_num, thread_num,
        *d_steps, *h_steps_old, *h_steps_new,
        step,
        i = 0,
        p = 1,
        number_of_leafs = (n + 1) / 2,
        h = (int)ceil(log2((float)n + 1));

    h_steps_new = (int *)malloc(number_of_leafs * 2 * sizeof(int));
    h_steps_old = (int *)malloc(number_of_leafs * 2 * sizeof(int));

    h_steps_new[0] = 0;
    h_steps_old[0] = 0;
    h_steps_old[1] = n;
    h_steps_new[1] = n;

    checkCudaErrors(
        cudaMalloc(&d_steps, number_of_leafs * 2 * sizeof(int)));
    checkCudaErrors(
        cudaMalloc(&d_partition, n * sizeof(int)));
    checkCudaErrors(
        cudaMalloc(&d_points, n * sizeof(Point)));
    checkCudaErrors(
        cudaMalloc(&d_swap, n * sizeof(Point)));

    checkCudaErrors(
        cudaMemcpy(d_points, h_points, n * sizeof(Point), cudaMemcpyHostToDevice));

    radixSelectAndPartition(d_points, d_swap, d_partition, n, dim);

    UpDim(dim);
    i++;
    while (i < (h - 1) )
    {
        nextStep(h_steps_new, h_steps_old, p <<= 1);
        step = h_steps_new[1] - h_steps_new[0];
        checkCudaErrors(
            cudaMemcpy(d_steps, h_steps_new, p * 2 * sizeof(int), cudaMemcpyHostToDevice));

        if (step >= 9000000)
        {
            singleRadixSelectAndPartition(d_points, d_swap, d_partition, h_steps_new, p, dim);
        }
        else if (step > 3000)
        {
            multiRadixSelectAndPartition(d_points, d_swap, d_partition, d_steps, step, p, dim);
        }
        else if (step > 3)
        {
            quickSelectAndPartition(d_points, d_steps, step, p, dim);
        }
        else
        {
            getThreadAndBlockCountForBuild(n, block_num, thread_num);
            balanceLeafs <<< block_num, thread_num >>> (d_points, d_steps, p, dim);
        }
        swap_pointer(&h_steps_new, &h_steps_old);
        i++;
        UpDim(dim);
    }

    checkCudaErrors(cudaFree(d_swap));
    checkCudaErrors(cudaFree(d_partition));
    checkCudaErrors(cudaFree(d_steps));
    free(h_steps_new);
    free(h_steps_old);

    checkCudaErrors(cudaMalloc(&d_tree, n * sizeof(Node)));

    getThreadAndBlockCountForBuild(n, block_num, thread_num);
    convertPoints <<< block_num, thread_num >>> (d_points, n, d_tree);

    checkCudaErrors(cudaMemcpy(tree, d_tree, n * sizeof(Node), cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(d_points));
    checkCudaErrors(cudaFree(d_tree));
}

void buildKdTreeStep(struct Point *h_points, int n, int dim, struct Node *tree)
{
    if (n <= 0) return;

    size_t free_bytes, needed_bytes;
    int m = n >> 1;

    free_bytes = getFreeBytesOnGpu_();
    needed_bytes = getNeededBytesForBuildingKdTree(n);

    if (free_bytes > needed_bytes)
    {
        cuBuildKdTree(h_points, n, dim, tree);
    }
    else
    {
        cpuQuickSelect(h_points, n, dim);
        pointConvert(tree[m], h_points[m]);

        UpDim(dim);

        buildKdTreeStep(h_points, m, dim, tree);
        buildKdTreeStep(h_points + m + 1, n - m - 1, dim, tree + m + 1);
    }
}


void buildKdTree(struct Point *h_points, int n, struct Node *tree)
{

    int dim = 0;
    buildKdTreeStep(h_points, n, dim, tree);
    store_locations(tree, 0, n, n);
}

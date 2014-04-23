#include <stdlib.h>
#include <math.h>
#include <float.h>

#include <helper_cuda.h>

#include "kd-search.cuh"

__device__
float cuDist(struct Point qp, struct Point point)
{
    float dx = qp.p[0] - point.p[0],
          dy = qp.p[1] - point.p[1],
          dz = qp.p[2] - point.p[2];

    return dx * dx + dy * dy + dz * dz;
}

__device__
void cuPush(int **stack, int value)
{
    *((*stack)++) = value;
}

__device__
void cuInitStack(int **stack)
{
    cuPush(stack, -1);
}

__device__
int cuPop(int **stack)
{
    return *(--(*stack));
}

__device__
int cuPeek(int *stack)
{
    return *(stack - 1);
}

__device__
bool cuIsNotEmpty(int *stack)
{
    return !(cuPeek(stack) == -1);
}

__device__
void cuInitKStack(KPoint **k_stack, int n)
{
    (*k_stack)[0].dist = -1;
    (*k_stack)++;
    for (int i = 0; i < n; ++i)
    {
        (*k_stack)[i].dist = FLT_MAX;
    }
}

__device__
void cuInsert(struct KPoint *k_stack, struct KPoint k_point, int n)
{
    int i = n - 1;
    KPoint swap;
    k_stack[n - 1].index = k_point.index;
    k_stack[n - 1].dist = k_point.dist;

    while (k_stack[i].dist < k_stack[i - 1].dist)
    {
        swap = k_stack[i], k_stack[i] = k_stack[i - 1], k_stack[i - 1] = swap;
        i--;
    }
}

__device__
struct KPoint cuLook(struct KPoint *k_stack, int n)
{
    return k_stack[n - 1];
}

__device__
int cuTarget(Point qp, Point current, int dim)
{
    if (qp.p[dim] <= current.p[dim])
    {
        return current.left;
    }
    return current.right;
}

__device__
int cuOther(Point qp, Point current, int dim)
{
    if (qp.p[dim] <= current.p[dim])
    {
        return current.right;
    }
    return current.left;
}

__device__
void cuUpDim(int *dim)
{
    *dim = (*dim + 1) % 3;
}

__device__
void cuKNN(struct Point qp, struct Point *tree, int n, int k, int *result, int *stack_ptr, int *d_stack_ptr, struct KPoint *k_stack_ptr)
{
    int *stack = stack_ptr,
         *d_stack = d_stack_ptr,
          dim = 2,
          current = n / 2;

    float current_dist;

    struct Point current_point;

    struct KPoint *k_stack = k_stack_ptr,
                           worst_best;

    worst_best.dist = FLT_MAX;

    cuInitStack(&stack);
    cuInitStack(&d_stack);
    cuInitKStack(&k_stack, k);

    while (cuIsNotEmpty(stack) || current != -1)
{
        if (current == -1 && cuIsNotEmpty(stack))
        {
            current = cuPop(&stack);
            current_point = tree[current];
            dim = cuPop(&d_stack);

            // printf("(%3.1f, %3.1f, %3.1f) current = %d dim = %d\n",
            // current_point.p[0], current_point.p[1], current_point.p[2], current, dim);

            current_dist = cuDist(qp, current_point);
            if (worst_best.dist > current_dist)
            {
                worst_best.dist = current_dist;
                worst_best.index = current;
                cuInsert(k_stack, worst_best, k);
                worst_best = cuLook(k_stack, k);
            }

            current = -1;
            if ((current_point.p[dim] - qp.p[dim]) * (current_point.p[dim] - qp.p[dim]) < worst_best.dist)
            {
                current = cuOther(qp, current_point, dim);
            }
        }
        else
        {
            cuUpDim(&dim);
            cuPush(&d_stack, dim);
            cuPush(&stack, current);
            current = cuTarget(qp, tree[current], dim);
        }
    }

    for (int i = 0; i < k; ++i)
    {
        result[i] = k_stack[i].index;
    }

}

template <int thread_stack_size>
__global__ void dQueryAll(struct Point *query_points, struct Point *tree, int n_qp, int n_tree, int k, int *result)
{
    int tid = threadIdx.x,
        rest = n_qp % gridDim.x,
        block_step = n_qp / gridDim.x,
        // *l_stack,
        block_offset = block_step * blockIdx.x;

    // __shared__ int stack[thread_stack_size * THREADS_PER_BLOCK_SEARCH];
    // l_stack = stack;
    // l_stack += thread_stack_size * threadIdx.x;

    if (rest >= gridDim.x - blockIdx.x)
    {
        block_offset += rest - (gridDim.x - blockIdx.x);
        block_step++;
    }

    query_points += block_offset;
    result += block_offset * k;

    int *stack_ptr = (int *)malloc(51 * sizeof(int)),
         *d_stack_ptr = (int *)malloc(51 * sizeof(int));

    struct KPoint *k_stack_ptr = (struct KPoint *) malloc((k + 1) * sizeof(KPoint));

    while (tid < block_step)
    {
        cuKNN(query_points[tid], tree, n_tree, k, result + (tid * k), stack_ptr, d_stack_ptr, k_stack_ptr);
        // printf("tid = %d, result = %d, query_point = %3.1f\n", tid, result[tid], query_points[tid].p[0]);
        tid += blockDim.x;
    }
    free(stack_ptr);
    free(d_stack_ptr);
    free(k_stack_ptr);
}

void getThreadAndBlockCountForQueryAll(int n, int &blocks, int &threads)
{
    threads = THREADS_PER_BLOCK_SEARCH;
    blocks = n / threads;
    blocks = min(MAX_BLOCK_DIM_SIZE, blocks);
    blocks = max(1, blocks);
    // printf("blocks = %d, threads = %d, n= %d\n", blocks, threads, n);
}

void queryAll(struct Point *h_query_points, struct Point *h_tree, int n_qp, int n_tree, int k, int *h_result)
{
    int *d_result, numBlocks, numThreads;
    struct Point *d_tree, *d_query_points;

    checkCudaErrors(cudaMalloc(&d_result, n_qp * k  * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_query_points, n_qp * sizeof(Point)));
    checkCudaErrors(cudaMalloc(&d_tree, n_tree * sizeof(Point)));

    checkCudaErrors(cudaMemcpy(d_query_points, h_query_points, n_qp * sizeof(Point), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_tree, h_tree, n_tree * sizeof(Point), cudaMemcpyHostToDevice));

    getThreadAndBlockCountForQueryAll(n_qp, numBlocks, numThreads);
    dQueryAll<50> <<< numBlocks, numThreads>>>(d_query_points, d_tree, n_qp, n_tree, k, d_result);

    checkCudaErrors(cudaMemcpy(h_result, d_result, n_qp * k * sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_tree));
    checkCudaErrors(cudaFree(d_query_points));
    checkCudaErrors(cudaFree(d_result));
}

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

    return (dx * dx) + (dy * dy) + (dz * dz);
}

__device__
void cuPush(struct SPoint **stack, struct SPoint value)
{
    *((*stack)++) = value;
}

__device__
void cuInitStack(struct SPoint **stack)
{
    struct SPoint temp;
    temp.index = -1;
    temp.dim = -1;
    cuPush(stack, temp);
}

__device__
struct SPoint cuPop(struct SPoint **stack)
{
    return *(--(*stack));
}

__device__
struct SPoint cuPeek(struct SPoint *stack)
{
    return *(stack - 1);
}

__device__
bool cuIsEmpty(struct SPoint *stack)
{
    return cuPeek(stack).index == -1;
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
int cuTarget(Point qp, Point current, float dx)
{
    if (dx > 0)
    {
        return current.left;
    }
    return current.right;
}

__device__
int cuOther(Point qp, Point current, float dx)
{
    if (dx > 0)
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
void cuKNN(struct Point qp, struct Point *tree, int n, int k, int *result,
         struct SPoint *stack_ptr, struct KPoint *k_stack_ptr)
{
    int  dim = 2;
    float current_dist, dx, dx2;

    struct Point current_point;
    struct SPoint *stack = stack_ptr,
                           current;
    struct KPoint *k_stack = k_stack_ptr,
                           worst_best;

    current.index = n / 2;
    worst_best.dist = FLT_MAX;

    cuInitStack(&stack);
    cuInitKStack(&k_stack, k);

    while (!cuIsEmpty(stack) || current.index != -1)
{
        if (current.index == -1 && !cuIsEmpty(stack))
        {
            current = cuPop(&stack);
            current_point = tree[current.index];
            dim = current.dim;



            dx = current_point.p[dim] - qp.p[dim];
            dx2 = dx * dx;

            // printf("Up with (%3.1f, %3.1f, %3.1f): best_dist = %3.1f, dx2 = %3.1f, dim = %d\n",
            //        current_point.p[0], current_point.p[1], current_point.p[2], worst_best.dist, dx2, dim);

            current.index = (dx2 < worst_best.dist) ? cuOther(qp, current_point, dx) : -1;
        }
        else
        {
            current_point = tree[current.index];

            current_dist = cuDist(qp, current_point);
            if (worst_best.dist > current_dist)
            {
                worst_best.dist = current_dist;
                worst_best.index = current.index;
                cuInsert(k_stack, worst_best, k);
                worst_best = cuLook(k_stack, k);
            }

            cuUpDim(&dim);
            current.dim = dim;
            cuPush(&stack, current);

            dx = current_point.p[dim] - qp.p[dim];
            current.index = cuTarget(qp, current_point, dx);

            // printf("Down with(%3.1f, %3.1f, %3.1f): best_dist = %3.1f, current_dist = %3.1f, dim = %d\n",
            //        current_point.p[0], current_point.p[1], current_point.p[2], worst_best.dist, current_dist, dim);
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

    struct SPoint *s_stack_ptr = (struct SPoint *)malloc(51 * sizeof(struct SPoint));
    struct KPoint *k_stack_ptr = (struct KPoint *) malloc((k + 1) * sizeof(KPoint));

    while (tid < block_step)
    {
        cuKNN(query_points[tid], tree, n_tree, k, result + (tid * k), s_stack_ptr, k_stack_ptr);
        // printf("tid = %d, result = %d, query_point = %3.1f\n", tid, result[tid], query_points[tid].p[0]);
        tid += blockDim.x;
    }
    
    free(s_stack_ptr);
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

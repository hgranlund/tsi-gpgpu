#include <stdlib.h>
#include <math.h>
#include <float.h>

#include <helper_cuda.h>

#include "kd-search.cuh"

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

__device__
float cuDist(Point qp, Point *points, int x)
{
    float dx = qp.p[0] - points[x].p[0],
          dy = qp.p[1] - points[x].p[1],
          dz = qp.p[2] - points[x].p[2];

    return dx * dx + dy * dy + dz * dz;
}

__device__
void cuPush(int *stack, int *eos, int value)
{
    (*eos)++;
    stack[*eos] = value;
}

__device__
int cuPop(int *stack, int *eos)
{
    if (*eos > -1)
    {
        (*eos)--;
        return stack[*eos + 1];
    }
    else
    {
        return -1;
    }
}

__device__
int cuPeek(int *stack, int eos)
{
    if (eos > -1)
    {
        return stack[eos];
    }
    else
    {
        return -1;
    }
}

__device__
void cuUpDim(int *dim)
{
    if (*dim >= 2)
    {
        (*dim) = 0;
    }
    else
    {
        (*dim)++;
    }
}

__device__
void cuDownDim(int *dim)
{
    if (*dim <= 0)
    {
        (*dim) = 2;
    }
    else
    {
        (*dim)--;
    }
}

__device__
int cuSearch(Point qp, Point *tree, int *stack, int n, int k)
{
    int eos = -1,
        // *stack = (int *) malloc(2 * log2((float) n) * sizeof stack),

        dim = 0,
        best = -1,

        previous = -1,
        current,
        target,
        other;

    float best_dist = FLT_MAX,
          current_dist;

    cuPush(stack, &eos, n / 2);
    cuUpDim(&dim);

    while (eos > -1)
    {
        current = cuPeek(stack, eos);
        target = tree[current].left;
        other = tree[current].right;

        current_dist = cuDist(qp, tree, current);

        if (current_dist < best_dist)
        {
            best_dist = current_dist;
            best = current;
        }

        if (qp.p[dim] > tree[current].p[dim])
        {
            int temp = target;

            target = other;
            other = temp;
        }

        if (previous == target)
        {
            if (other > -1 && (tree[current].p[dim] - qp.p[dim]) * (tree[current].p[dim] - qp.p[dim]) < best_dist)
            {
                cuPush(stack, &eos, other);
                cuUpDim(&dim);
            }
            else
            {
                cuPop(stack, &eos);
                cuDownDim(&dim);
            }
        }
        else if (previous == other)
        {
            cuPop(stack, &eos);
            cuDownDim(&dim);
        }
        else
        {
            if (target > -1)
            {
                cuPush(stack, &eos, target);
                cuUpDim(&dim);
            }
            else if (other > -1)
            {
                cuPush(stack, &eos, other);
                cuUpDim(&dim);
            }
            else
            {
                cuPop(stack, &eos);
                cuDownDim(&dim);
            }
        }
        previous = current;
    }
    return best;
}

// __device__
// int cuSearch(Point qp, Point *tree, int *stack, int n, int k)
// {
//     printf("block/thread = %2d/%2d, qp = (%4.0f, %4.0f, %4.0f), n= %2d, k = %2d\n", blockIdx.x, threadIdx.x, qp.p[0], qp.p[1], qp.p[2], n, k );
//     return 0;
// }

template <int thread_stack_size>
__global__
void dQueryAll(Point *query_points, Point *tree, int n_qp, int n_tree, int k, int *result)
{
    int tid = threadIdx.x,
        rest = n_qp % gridDim.x,
        block_step = n_qp / gridDim.x,
        *l_stack,
        block_offset = block_step * blockIdx.x;

    __shared__ int stack[thread_stack_size * THREADS_PER_BLOCK_SEARCH];
    l_stack = stack;
    l_stack += thread_stack_size * threadIdx.x;

    if (rest >= gridDim.x - blockIdx.x)
    {
        block_offset += rest - (gridDim.x - blockIdx.x);
        block_step++;
    }
    // if (tid == 0)
    // {
    //     printf("block/tread = %d/%d, block_offset = %d, blockStep = %d, bloks/threads = %d/%d\n", blockIdx.x, threadIdx.x, block_offset, block_step, gridDim.x,
    //            blockDim.x  );
    // }
    query_points += block_offset;
    result += block_offset * k;
    while (tid < block_step)
    {

        result[tid] = cuSearch(query_points[tid], tree, stack, n_tree, k);
        // printf("tid = %d, result = %d, query_point = %3.1f\n", tid, result[tid], query_points[tid].p[0]);
        tid += blockDim.x;
    }
}

void getThreadAndBlockCountForQueryAll(int n, int &blocks, int &threads)
{
    threads = THREADS_PER_BLOCK_SEARCH;
    blocks = n / threads;
    blocks = min(MAX_BLOCK_DIM_SIZE, blocks);
    blocks = max(1, blocks);
    // printf("blocks = %d, threads = %d, n= %d\n", blocks, threads, n);
}

void queryAll(Point *h_query_points, Point *h_tree, int n_qp, int n_tree, int k, int *h_result)
{
    int *d_result, numBlocks, numThreads;
    Point *d_tree, *d_query_points;

    // printf("query points :[");
    // for (int i = 0; i < n_qp; ++i)
    // {
    //     printf("(%3.1f, %3.1f, %3.1f)\n", h_query_points[i].p[0], h_query_points[i].p[1], h_query_points[i].p[2]);
    // }
    // printf("\n" );
    checkCudaErrors(cudaMalloc(&d_result, n_qp * k  * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_query_points, n_qp * sizeof(Point)));
    checkCudaErrors(cudaMalloc(&d_tree, n_tree * sizeof(Point)));

    checkCudaErrors(cudaMemcpy(d_query_points, h_query_points, n_qp * sizeof(Point), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_tree, h_tree, n_tree * sizeof(Point), cudaMemcpyHostToDevice));

    getThreadAndBlockCountForQueryAll(n_qp, numBlocks, numThreads);
    dQueryAll<150> <<< numBlocks, numThreads>>>(d_query_points, d_tree, n_qp, n_tree, k, d_result);

    checkCudaErrors(cudaMemcpy(h_result, d_result, n_qp * k * sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_tree));
    checkCudaErrors(cudaFree(d_query_points));
    checkCudaErrors(cudaFree(d_result));
}

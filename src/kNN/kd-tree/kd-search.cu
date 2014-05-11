#include <stdlib.h>
#include <math.h>
#include <float.h>

#include <helper_cuda.h>
#include <kd-search.cuh>

__device__ __host__
float cuDist(struct Point qp, struct Node point)
{
    float dx = qp.p[0] - point.p[0],
          dy = qp.p[1] - point.p[1],
          dz = qp.p[2] - point.p[2];

    return (dx * dx) + (dy * dy) + (dz * dz);
}

__device__ __host__
void cuInitStack(struct SPoint **stack)
{
    struct SPoint temp;
    temp.index = -1;
    temp.dim = -1;
    cuPush(stack, temp);
}

__device__ __host__
bool cuIsEmpty(struct SPoint *stack)
{
    return cuPeek(stack).index == -1;
}

__device__ __host__
void cuPush(struct SPoint **stack, struct SPoint value)
{
    *((*stack)++) = value;
}

__device__ __host__
struct SPoint cuPop(struct SPoint **stack)
{
    return *(--(*stack));
}

__device__ __host__
struct SPoint cuPeek(struct SPoint *stack)
{
    return *(stack - 1);
}


__device__ __host__
void cuInitKStack(struct KPoint **k_stack, int n)
{
    (*k_stack)[0].dist = -1;
    (*k_stack)++;
    for (int i = 0; i < n; ++i)
    {
        (*k_stack)[i].dist = FLT_MAX;
    }
}

__device__ __host__
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

__device__ __host__
struct KPoint cuLook(struct KPoint *k_stack, int n)
{
    return k_stack[n - 1];
}

__device__ __host__
void cuUpDim(int *dim)
{
    *dim = (*dim + 1) % 3;
}

__device__ __host__
void cuChildren(struct Point qp, struct Node current, float dx, int &target, int &other)
{
    if (dx > 0)
    {
        other = current.right;
        target = current.left;
    }
    else
    {
        other = current.left;
        target = current.right;
    }
}

__device__ __host__
void cuKNN(struct Point qp, struct Node *tree, int n, int k, int *result,
           struct SPoint *stack_ptr, struct KPoint *k_stack_ptr)
{
    int  dim = 2, target;
    float current_dist;

    struct Node current_point;
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
            dim = current.dim;

            current.index = (current.dx * current.dx < worst_best.dist) ? current.other : -1;
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
            current.dx = current_point.p[dim] - qp.p[dim];
            cuChildren(qp, current_point, current.dx, target, current.other);
            cuPush(&stack, current);

            current.index = target;
        }
    }

    for (int i = 0; i < k; ++i)
    {
        result[i] = k_stack[i].index;
    }
}

__device__ __host__ int fastIntegerLog2(int x)
{
    int y = 0;
    while (x > 2)
    {
        y++;
        x >>= 1;
    }
    return y;
}

__device__ void cuCalculateBlockOffsetAndNoOfQueries(int n, int &n_per_block, int &block_offset)
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

size_t getFreeBytesOnGpu()
{
    size_t free_byte, total_byte ;
    cudaError_t cuda_status = cudaMemGetInfo( &free_byte, &total_byte ) ;
    return free_byte - 64;
}
size_t getNeededBytes(int n_qp, int k, int n, int thread_num, int block_num)
{
    return n_qp * (k * sizeof(int) +  sizeof(Point)) +
           (thread_num * block_num *
            ( (k + 1) * sizeof(KPoint) + (fastIntegerLog2(n) * 2 * sizeof(SPoint))));
}

__global__
void dQueryAll(struct Point *query_points, struct Node *tree, int n_qp, int n_tree, int k, int *result)
{
    int tid = threadIdx.x,
        stack_size = fastIntegerLog2(n_tree) * 2,
        block_step,
        block_offset;

    struct KPoint *k_stack_ptr = (struct KPoint *) malloc((k + 1) * sizeof(KPoint));
    struct SPoint *s_stack_ptr = (struct SPoint *) malloc((stack_size) * sizeof(SPoint));

    cuCalculateBlockOffsetAndNoOfQueries(n_qp, block_step, block_offset);

    query_points += block_offset;
    result += block_offset * k;

    while (tid < block_step)
    {
        cuKNN(query_points[tid], tree, n_tree, k, result + (tid * k), s_stack_ptr, k_stack_ptr);
        tid += blockDim.x;
    }

    free(k_stack_ptr);
    free(s_stack_ptr);
}

void getThreadAndBlockCountForQueryAll(int n, int &blocks, int &threads)
{
    threads = THREADS_PER_BLOCK_SEARCH;
    blocks = n / threads;
    blocks = min(MAX_BLOCK_DIM_SIZE, blocks);
    blocks = max(1, blocks);
    // printf("blocks = %d, threads = %d, n= %d\n", blocks, threads, n);
}

int getQueriesInStep(int n_qp, int k, int n)
{
    int numBlocks, numThreads;
    size_t needed_bytes_total, free_bytes;

    free_bytes = getFreeBytesOnGpu();

    getThreadAndBlockCountForQueryAll(n_qp, numThreads, numBlocks);
    needed_bytes_total = getNeededBytes(n_qp, k, n, numThreads, numBlocks);

    if (free_bytes < needed_bytes_total)
    {
        return getQueriesInStep(n_qp / 2, k, n);
    }
    else
    {
        return n_qp;
    }
}

void cuQueryAll(struct Point *h_query_points, struct Node *h_tree, int n_qp, int n_tree, int k, int *h_result)
{
    int *d_result, numBlocks, numThreads, queries_in_step, queries_done;
    struct Node *d_tree;
    struct Point *d_query_points;

    checkCudaErrors(cudaMalloc(&d_tree, n_tree * sizeof(Node)));
    checkCudaErrors(cudaMemcpy(d_tree, h_tree, n_tree * sizeof(Node), cudaMemcpyHostToDevice));

    queries_in_step = getQueriesInStep(n_qp, k, n_tree);
    queries_done = 0;

    checkCudaErrors(cudaMalloc(&d_result, queries_in_step * k  * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_query_points, queries_in_step * sizeof(Point)));

    while (queries_done < n_qp)
    {
        if (queries_done + queries_in_step > n_qp)
        {
            queries_in_step = n_qp - queries_done;
        }

        checkCudaErrors(cudaMemcpy(d_query_points, h_query_points, queries_in_step * sizeof(Point), cudaMemcpyHostToDevice));

        getThreadAndBlockCountForQueryAll(queries_in_step, numBlocks, numThreads);
        dQueryAll <<< numBlocks, numThreads>>>(d_query_points, d_tree, queries_in_step, n_tree, k, d_result);

        checkCudaErrors(cudaMemcpy(h_result, d_result, queries_in_step * k * sizeof(int), cudaMemcpyDeviceToHost));

        h_query_points += queries_in_step;
        h_result += queries_in_step;
        queries_done += queries_in_step;
    }

    checkCudaErrors(cudaFree(d_query_points));
    checkCudaErrors(cudaFree(d_result));
    checkCudaErrors(cudaFree(d_tree));
}

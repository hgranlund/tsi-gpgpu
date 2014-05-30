#include <stdlib.h>
#include <math.h>
#include <float.h>

#include <helper_cuda.h>
#include <cu-kd-search.cuh>
#include "kd-tree-build.cuh"

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
    (*stack)[0].index = -1;
    (*stack)++;
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
    (*k_stack)--;
    for (int i = 1; i <= n; ++i)
    {
        (*k_stack)[i].dist = FLT_MAX;
        (*k_stack)[i].index = -1;
    }
}

__device__ __host__
void cuInsert(struct KPoint *k_stack, struct KPoint k_point, int n)
{
    int i_child, now;
    struct KPoint child, child_tmp_2;
    for (now = 1; now * 2 <= n ; now = i_child)
    {
        i_child = now * 2;
        child = k_stack[i_child];
        child_tmp_2 = k_stack[i_child + 1];
        if (i_child <= n && child_tmp_2.dist > child.dist )
        {
            i_child++;
            child = child_tmp_2;
        }

        if (i_child <= n && k_point.dist < child.dist)
        {
            k_stack[now] = child;
        }
        else
        {
            break;
        }
    }
    k_stack[now] = k_point;
}


__device__ __host__
struct KPoint cuLook(struct KPoint *k_stack)
{
    return k_stack[1];
}

__device__ __host__
void cuUpDim(int &dim)
{
    dim = (dim + 1) % 3;
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
void cuKNN(struct Point qp, struct Node *tree, int n, int k,
           struct SPoint *stack, struct KPoint *k_stack)
{
    int  dim = 2, target;
    float current_dist;

    struct Node current_point;
    struct SPoint current;
    struct KPoint worst_best;

    current.index = n / 2;

    cuInitStack(&stack);
    cuInitKStack(&k_stack, k);
    worst_best = cuLook(k_stack);

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
                worst_best = cuLook(k_stack);
            }

            cuUpDim(dim);
            current.dim = dim;
            current.dx = current_point.p[dim] - qp.p[dim];
            cuChildren(qp, current_point, current.dx, target, current.other);
            cuPush(&stack, current);

            current.index = target;
        }
    }
}

__device__ __host__
int fastIntegerLog2(int x)
{
    int y = 0;
    while (x >>= 1)
    {
        y++;
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

__device__ __host__ int getSStackSize(int n)
{
    return fastIntegerLog2(n) + 2;
}

size_t getSStackSizeInBytes(int n, int thread_num, int block_num)
{
    return block_num * thread_num * ((getSStackSize(n) * sizeof(SPoint)));
}

size_t getNeededBytesInSearch(int n_qp, int k, int n, int thread_num, int block_num)
{
    return n_qp * (k * sizeof(int) +  sizeof(Point)) +
           (n_qp * k * sizeof(KPoint))  +
           (getSStackSizeInBytes(n, thread_num, block_num));
}

void populateTrivialResult(int n_qp, int k, int n_tree, int *result)
{
    #pragma omp parallel for
    for (int i = 0; i < n_qp; ++i)
    {
        for (int j = 0; j < k; ++j)
        {
            result[i * k + j] = j % n_tree;
        }
    }
}

template<int stack_size> __global__
void dQueryAll(struct Point *query_points, struct Node *tree, int n_qp, int n_tree, int k, struct KPoint *k_stack_ptr)
{

    SPoint stack[stack_size];

    int tid = threadIdx.x,
        block_step,
        block_offset;

    cuCalculateBlockOffsetAndNoOfQueries(n_qp, block_step, block_offset);

    query_points += block_offset;
    k_stack_ptr += block_offset * k;

    while (tid < block_step)
    {
        cuKNN(query_points[tid], tree, n_tree, k, stack, k_stack_ptr + (tid * k));
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

int getQueriesInStep(int n_qp, int k, int n)
{
    int numBlocks, numThreads;
    size_t needed_bytes_total, free_bytes;

    free_bytes = getFreeBytesOnGpu();

    getThreadAndBlockCountForQueryAll(n_qp, numThreads, numBlocks);

    needed_bytes_total = getNeededBytesInSearch(n_qp, k, n, numThreads, numBlocks);

    if (free_bytes > needed_bytes_total) return n_qp;
    if (n_qp < 50) return -1;

    return getQueriesInStep((n_qp / 2), k, n);
}

int nextPowerOf2___(int x)
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

void templateQueryAll(struct Point *d_query_points, struct Node *d_tree, int queries_in_step, int n_tree, int k, int stack_size, int numBlocks, int numThreads, struct KPoint *d_k_stack)
{
    if (stack_size <= 20 )
    {
        dQueryAll<20> <<< numBlocks, numThreads>>>(d_query_points, d_tree, queries_in_step, n_tree, k, d_k_stack);
    }
    else if (stack_size <= 25)
    {
        dQueryAll<25> <<< numBlocks, numThreads>>>(d_query_points, d_tree, queries_in_step, n_tree, k, d_k_stack);
    }
    else if (stack_size <= 30)
    {
        dQueryAll<30> <<< numBlocks, numThreads>>>(d_query_points, d_tree, queries_in_step, n_tree, k, d_k_stack);
    }
    else
    {
        dQueryAll<35> <<< numBlocks, numThreads>>>(d_query_points, d_tree, queries_in_step, n_tree, k, d_k_stack);
    }
}


void cuQueryAll(struct Point *h_query_points, struct Node *h_tree, int n_qp, int n_tree, int k, int *h_result)
{
    int numBlocks, numThreads, queries_in_step, queries_done, stack_size;
    struct Node *d_tree;
    struct KPoint *d_k_stack, *h_k_stack;
    struct SPoint *d_stack;
    struct Point *d_query_points;

    if (k >= n_tree)
    {
        populateTrivialResult(n_qp, k, n_tree, h_result);
        return;
    }

    checkCudaErrors(cudaMalloc(&d_tree, n_tree * sizeof(Node)));
    checkCudaErrors(cudaMemcpy(d_tree, h_tree, n_tree * sizeof(Node), cudaMemcpyHostToDevice));

    queries_in_step = getQueriesInStep(n_qp, k, n_tree);

    if (queries_in_step <= 0)
    {
        printf("There is not enough memory to perform this queries on cuda.\n");
        return;
    }

    queries_done = 0;
    stack_size = getSStackSize(n_tree);
    queries_in_step = nextPowerOf2___(++queries_in_step) >> 1;
    getThreadAndBlockCountForQueryAll(queries_in_step, numBlocks, numThreads);

    h_k_stack = (KPoint *) malloc(queries_in_step * k * sizeof(KPoint));
    checkCudaErrors(cudaMalloc(&d_k_stack, queries_in_step * k  * sizeof(KPoint)));
    checkCudaErrors(cudaMalloc(&d_stack, numThreads * numBlocks * stack_size * sizeof(SPoint)));
    checkCudaErrors(cudaMalloc(&d_query_points, queries_in_step * sizeof(Point)));

    while (queries_done < n_qp)
    {
        if (queries_done + queries_in_step > n_qp)
        {
            queries_in_step = n_qp - queries_done;
        }
        checkCudaErrors(cudaMemcpy(d_query_points, h_query_points, queries_in_step * sizeof(Point), cudaMemcpyHostToDevice));

        templateQueryAll(d_query_points, d_tree, queries_in_step, n_tree, k, stack_size, numBlocks, numThreads, d_k_stack);

        checkCudaErrors(cudaMemcpy(h_k_stack, d_k_stack, queries_in_step * k * sizeof(KPoint), cudaMemcpyDeviceToHost));

        # pragma omp parallel for
        for (int i = 0; i < queries_in_step ; ++i)
        {
            for (int j = 0; j < k; ++j)
            {
                h_result[i * k + j] = h_k_stack[i * k + j].index;
            }
        }

        h_query_points += queries_in_step;
        h_result += (queries_in_step * k);
        queries_done += queries_in_step;
    }

    free(h_k_stack);
    checkCudaErrors(cudaFree(d_query_points));
    checkCudaErrors(cudaFree(d_k_stack));
    checkCudaErrors(cudaFree(d_tree));
}

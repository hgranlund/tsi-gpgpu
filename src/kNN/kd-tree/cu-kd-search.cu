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
void cuKNN(struct Point qp, struct Node *tree, int n, int k,
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
}

__device__ __host__
int fastIntegerLog2(int x)
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

__device__ __host__ int getSStackSize(int n)
{
    return fastIntegerLog2(n) + 5;
}

size_t getFreeBytesOnGpu()
{
    size_t free_byte, total_byte ;
    cudaError_t cuda_status = cudaMemGetInfo( &free_byte, &total_byte ) ;
    return free_byte - 1024;
}

size_t getNeededHeapSize(int n, int k, int thread_num, int block_num)
{
    return block_num * thread_num * ((getSStackSize(n) * sizeof(SPoint)));
}

size_t getNeededBytesInSearch(int n_qp, int k, int n, int thread_num, int block_num)
{
    return n_qp * (k * sizeof(int) +  sizeof(Point)) +
           (n_qp * (k + 1) * sizeof(KPoint))  +
           (getNeededHeapSize(n, k, thread_num, block_num));
}

void setHeapSize(int queries_in_step, int k, int n_tree, int numThreads, int numBlocks)
{
    size_t heap_size, needed_heap_size;

    needed_heap_size = getNeededHeapSize(n_tree, k, numThreads, numBlocks);

    checkCudaErrors(
        cudaDeviceGetLimit(&heap_size, cudaLimitMallocHeapSize));

    if (needed_heap_size < heap_size )
    {
        checkCudaErrors(
            cudaDeviceSetLimit(cudaLimitMallocHeapSize, needed_heap_size));
    }
}

void populateTrivialResult(int n_qp, int k, int n_tree, int *result)
{
    int i , j;
    for (i = 0; i < n_qp; ++i)
    {
        for (j = 0; j < k; ++j)
        {
            result[i * k + j] = j % n_tree;
        }
    }
}

__global__
void dQueryAll(struct Point *query_points, struct Node *tree, int n_qp, int n_tree, int k, struct KPoint *k_stack_ptr)
{

    int tid = threadIdx.x,
        stack_size = getSStackSize(n_tree),
        block_step,
        block_offset;

    struct SPoint *s_stack_ptr = (struct SPoint *) malloc((stack_size) * sizeof(SPoint));

    cuCalculateBlockOffsetAndNoOfQueries(n_qp, block_step, block_offset);

    query_points += block_offset;
    k_stack_ptr += block_offset * (k + 1);

    while (tid < block_step)
    {
        cuKNN(query_points[tid], tree, n_tree, k, s_stack_ptr, k_stack_ptr + (tid * (k + 1)));
        tid += blockDim.x;
    }

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
    needed_bytes_total = getNeededBytesInSearch(n_qp, k, n, numThreads, numBlocks);

    if (free_bytes > needed_bytes_total) return n_qp;
    if (n_qp < 50) return -1;

    return getQueriesInStep((n_qp * 4) / 5, k, n);
}

void cuQueryAll(struct Point *h_query_points, struct Node *h_tree, int n_qp, int n_tree, int k, int *h_result)
{
    int numBlocks, numThreads, queries_in_step, queries_done;
    struct Node *d_tree;
    struct KPoint *d_k_stack, *h_k_stack;
    struct Point *d_query_points;

    if (k >= n_tree)
    {
        populateTrivialResult(n_qp, k, n_tree, h_result);
        return;
    }


    checkCudaErrors(cudaMalloc(&d_tree, n_tree * sizeof(Node)));
    checkCudaErrors(cudaMemcpy(d_tree, h_tree, n_tree * sizeof(Node), cudaMemcpyHostToDevice));

    queries_in_step = getQueriesInStep(n_qp, k, n_tree);
    queries_done = 0;

    if (queries_in_step <= 0)
    {
        printf("There is not enough memory to perform this queries on cuda.\n");
        return;
    }

    getThreadAndBlockCountForQueryAll(queries_in_step, numBlocks, numThreads);

    setHeapSize(queries_in_step, k, n_tree, numThreads, numBlocks);

    h_k_stack = (KPoint *) malloc(queries_in_step * (k + 1) * sizeof(KPoint));
    checkCudaErrors(cudaMalloc(&d_k_stack, queries_in_step * (k + 1)  * sizeof(KPoint)));
    checkCudaErrors(cudaMalloc(&d_query_points, queries_in_step * sizeof(Point)));

    while (queries_done < n_qp)
    {
        if (queries_done + queries_in_step > n_qp)
        {
            queries_in_step = n_qp - queries_done;
        }
        checkCudaErrors(cudaMemcpy(d_query_points, h_query_points, queries_in_step * sizeof(Point), cudaMemcpyHostToDevice));

        dQueryAll <<< numBlocks, numThreads>>>(d_query_points, d_tree, queries_in_step, n_tree, k, d_k_stack);

        checkCudaErrors(cudaMemcpy(h_k_stack, d_k_stack, queries_in_step * (k + 1) * sizeof(KPoint), cudaMemcpyDeviceToHost));

        for (int i = 0; i < queries_in_step ; ++i)
        {
            for (int j = 0; j < k; ++j)
            {
                h_result[i * k + j] = h_k_stack[i * (k + 1) + 1].index;
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

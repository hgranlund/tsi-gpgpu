#include <stdlib.h>
#include <math.h>
#include <float.h>

#include <helper_cuda.h>

#include "kd-search.cuh"



__device__
float cuDist(struct Point qp, struct Point *points, int x)
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
int cuSearch(struct Point qp, struct Point *tree, int *stack, int n, int k)
{
    int eos = -1,
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

template <int thread_stack_size>
__global__ void dQueryAll(struct Point *query_points, struct Point *tree, int n_qp, int n_tree, int k, int *result)
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

    query_points += block_offset;
    result += block_offset * k;
    while (tid < block_step)
    {
        result[tid] = cuSearch(query_points[tid], tree, l_stack, n_tree, k);
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

    // show memory usage of GPU

    size_t free_byte ;

    size_t total_byte ;

    cudaError_t cuda_status = cudaMemGetInfo( &free_byte, &total_byte ) ;

    if ( cudaSuccess != cuda_status )
    {

        printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status) );

        exit(1);

    }



    double free_db = (double)free_byte ;

    double total_db = (double)total_byte ;

    double used_db = total_db - free_db ;

    printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n", used_db / 1024.0 / 1024.0, free_db / 1024.0 / 1024.0, total_db / 1024.0 / 1024.0);




    checkCudaErrors(cudaMemcpy(h_result, d_result, n_qp * k * sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_tree));
    checkCudaErrors(cudaFree(d_query_points));
    checkCudaErrors(cudaFree(d_result));
}

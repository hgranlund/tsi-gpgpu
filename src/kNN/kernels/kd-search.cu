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

// __device__
// float dist(Point qp, Point *points, int x)
// {
//     float dx = qp.p[0] - points[x].p[0],
//           dy = qp.p[1] - points[x].p[1],
//           dz = qp.p[2] - points[x].p[2];

//     return dx * dx + dy * dy + dz * dz;
// }

// __device__
// void push(int *stack, int *eos, int value)
// {
//     (*eos)++;
//     stack[*eos] = value;
// }

// __device__
// int pop(int *stack, int *eos)
// {
//     if (*eos > -1)
//     {
//         (*eos)--;
//         return stack[*eos + 1];
//     }
//     else
//     {
//         return -1;
//     }
// }

// __device__
// int peek(int *stack, int eos)
// {
//     if (eos > -1)
//     {
//         return stack[eos];
//     }
//     else
//     {
//         return -1;
//     }
// }

// __device__
// void upDim(int *dim)
// {
//     if (*dim >= 2)
//     {
//         (*dim) = 0;
//     }
//     else
//     {
//         (*dim)++;
//     }
// }

// __device__
// void downDim(int *dim)
// {
//     if (*dim <= 0)
//     {
//         (*dim) = 2;
//     }
//     else
//     {
//         (*dim)--;
//     }
// }

// __device__
// int nn(Point qp, Point *tree, int *stack, int n, int k)
// {
//     int eos = -1,
//         // *stack = (int *) malloc(2 * log2((float) n) * sizeof stack),

//          dim = 0,
//          best,

//          previous = -1,
//          current,
//          target,
//          other;

//     float best_dist = FLT_MAX,
//           current_dist;

//     push(stack, &eos, n / 2);
//     upDim(&dim);

//     while (eos > -1)
//     {
//         current = peek(stack, eos);
//         target = tree[current].left;
//         other = tree[current].right;

//         current_dist = dist(qp, tree, current);

//         if (current_dist < best_dist)
//         {
//             best_dist = current_dist;
//             best = current;
//         }

//         if (qp.p[dim] > tree[current].p[dim])
//         {
//             int temp = target;

//             target = other;
//             other = temp;
//         }

//         if (previous == target)
//         {
//             if (other > -1 && (tree[current].p[dim] - qp.p[dim]) * (tree[current].p[dim] - qp.p[dim]) < best_dist)
//             {
//                 push(stack, &eos, other);
//                 upDim(&dim);
//             }
//             else
//             {
//                 pop(stack, &eos);
//             }
//         }
//         else if (previous == other)
//         {
//             pop(stack, &eos);
//         }
//         else
//         {
//             if (target > -1)
//             {
//                 push(stack, &eos, target);
//                 upDim(&dim);
//             }
//             else if (other > -1)
//             {
//                 push(stack, &eos, other);
//                 upDim(&dim);
//             }
//             else
//             {
//                 pop(stack, &eos);
//             }
//         }

//         previous = current;
//     }

//     return best;
// }

__device__
int nn(Point qp, Point *tree, int n, int k)
{
    return 0;
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

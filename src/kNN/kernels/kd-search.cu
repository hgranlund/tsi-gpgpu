#include <stdlib.h>
#include <math.h>

#include <helper_cuda.h>

#include <kd-search.cuh>

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
float dist(float *qp, Point *points, int x)
{
    float dx = qp[0] - points[x].p[0],
        dy = qp[1] - points[x].p[1],
        dz = qp[2] - points[x].p[2];

    return dx*dx + dy*dy + dz*dz;
}

__device__
int nn(float *qp, Point *tree, int dim, int index)
{
    if (tree[index].left == -1 && tree[index].right == -1)
    {
        return index;
    }

    int target, other, d = dim % 3,
    
        target_index = tree[index].right,
        other_index = tree[index].left;
    
    dim++;

    if (tree[index].p[d] > qp[d] || target_index == -1)
    {
        int temp = target_index;

        target_index = other_index;
        other_index = temp;
    }

    target = nn(qp, tree, dim, target_index);
    float target_dist = dist(qp, tree, target);
    float current_dist = dist(qp, tree, index);

    if (current_dist < target_dist)
    {
        target_dist = current_dist;
        target = index;
    }

    if ((tree[index].p[d] - qp[d])*(tree[index].p[d] - qp[d]) > target_dist || other_index == -1)
    {
        return target;
    }

    other = nn(qp, tree, dim, other_index);
    float other_distance = dist(qp, tree, other);

    if (other_distance > target_dist)
    {
        return target;
    }
    return other;
}

__global__
void d_all_nearest(Point *tree, int n, int mid, int step)
{
    int i, result;
    float qp[3] = {0.0, 1.0, 2.0};

    // step = n / (gridDim.x * blockDim.x);
    // printf("blockIdx: %d, threadIdx: %d, gridDim: %d, blockDim: %d\n", blockIdx.x, threadIdx.x, gridDim.x, blockDim.x);

    result = nn(qp, tree, n, mid);

    // for (i = 0; i < step; ++i)
    // {
    //     result = nn(qp, tree, n, mid);
    // }
}

void all_nearest(Point *h_query_points, Point *h_tree, int qp_n, int tree_n)
{
    int numBlocks, numThreads, mid, step;
    Point *d_tree;

    // numBlocks = 1000;
    // numThreads = 100;

    numBlocks = 1;
    numThreads = 1;

    mid = (int) floor(tree_n / 2);
    step = ceil(tree_n / (numBlocks * numThreads));

    printf("Block size: %d, step: %d, n: %d\n", numBlocks, step, tree_n);

    checkCudaErrors(cudaMalloc(&d_tree, tree_n * sizeof(Point)));
    checkCudaErrors(cudaMemcpy(d_tree, h_tree, tree_n * sizeof(Point), cudaMemcpyHostToDevice));

    d_all_nearest<<<numBlocks,numThreads>>>(d_tree, tree_n, mid, step);

    checkCudaErrors(cudaFree(d_tree));
}

// int nn(float *qp, struct Point *tree, float *dists, int dim, int index)
// {
//     float current_dist = dist(qp, tree, index);
//     dists[index] = current_dist;

//     if (tree[index].left == -1 && tree[index].right == -1)
//     {
//         return index;
//     }

//     int target, other, d = dim % 3,
    
//         target_index = tree[index].right,
//         other_index = tree[index].left;
    
//     dim++;

//     if (tree[index].p[d] > qp[d] || target_index == -1)
//     {
//         int temp = target_index;

//         target_index = other_index;
//         other_index = temp;
//     }

//     target = nn(qp, tree, dists, dim, target_index);
//     float target_dist = dists[target];

//     if (current_dist < target_dist)
//     {
//         target_dist = current_dist;
//         target = index;
//     }

//     if ((tree[index].p[d] - qp[d])*(tree[index].p[d] - qp[d]) > target_dist || other_index == -1)
//     {
//         return target;
//     }

//     other = nn(qp, tree, dists, dim, other_index);
//     float other_distance = dists[other];

//     if (other_distance > target_dist)
//     {
//         return target;
//     }
//     return other;
// }
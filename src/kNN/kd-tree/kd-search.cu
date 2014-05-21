#include <stdlib.h>
#include <math.h>
#include <float.h>

#include <helper_cuda.h>
#include <kd-search.cuh>
#include <cu-kd-search.cuh>
#include "kd-tree-build.cuh"

size_t getTreeSize(int n)
{
    return n * sizeof(struct Node);
}

size_t getRequierdSizeForQueryAll(int n_qp, int k, int n)
{
    int numBlocks, numThreads;
    getThreadAndBlockCountForQueryAll(n, numBlocks, numThreads);
    return getTreeSize(n) + getNeededBytesInSearch(MIN_NUM_QUERY_POINTS, k, n, numThreads, numBlocks);
}

void maxHeapResultInsert(int *result, struct Node *tree, int point_index, struct Point qp, int k)
{
    int child, now;
    struct Node insert_node = tree[point_index];

    if (cuDist(qp, insert_node) > cuDist(qp, tree[result[0]]) ) return;

    for (now = 1; now * 2 <= k ; now = child)
    {
        child = now * 2;
        if (child <= k && cuDist(qp, tree[result[child + 1]]) > cuDist(qp, tree[result[child]]) )  child++;

        if (child <= k && cuDist(qp, insert_node) < cuDist(qp, tree[result[child]]))
        {
            result[now] = result[child];
        }
        else
        {
            break;
        }
    }
    result[now] = point_index;
}

void mergeResult(struct Node *tree, struct Point *query_points, int k, int n_qp, int root, int *result_right, int *result)
{
    #pragma omp parallel for
    for (int i_qp = 0; i_qp < n_qp; ++i_qp)
    {
        int i_k;
        int *result_max_heap = result + i_qp * k - 1;
        for (i_k = 0; i_k < k; ++i_k)
        {
            maxHeapResultInsert(result_max_heap, tree, result_right[i_qp * k + i_k], query_points[i_qp], k);
        }
    }
}

void queryAll(struct Point *h_query_points, struct Node *h_tree, int n_qp, int n_tree, int k, int *h_result, int switch_limit)
{
    if (getFreeBytesOnGpu() > getRequierdSizeForQueryAll(n_qp, k , n_tree) || k > n_tree)
    {
        cuQueryAll(h_query_points, h_tree, n_qp, n_tree, k, h_result);
    }
    else
    {
        struct Node *tree_rigth;
        int *h_result_right, root = n_tree / 2, n_tree_rigth;
        h_result_right = (int *) malloc(k * n_qp * sizeof(int));

        queryAll(h_query_points, h_tree, n_qp, root, k, h_result, switch_limit );
        cudaDeviceReset();

        n_tree_rigth = n_tree - root - 1;
        tree_rigth = h_tree + (root + 1);
        store_locations(tree_rigth, 0, n_tree_rigth, n_tree_rigth);

        queryAll(h_query_points, tree_rigth, n_qp, n_tree_rigth, k, h_result_right, switch_limit );

        mergeResult(h_tree, h_query_points, k, n_qp, root, h_result_right, h_result);
    }
}




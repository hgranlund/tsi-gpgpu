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

void mergeResult(struct Node *tree, struct Point *query_points, int k, int n_qp, int root, int *h_result_right, int *h_result)
{

    int i_qp, i_k, i_r, i_l, *merge_list_left, *merge_list_right;
    struct Point qp;
    merge_list_left = (int *) malloc(k * sizeof(int));

    for (i_qp = 0; i_qp < n_qp; ++i_qp)
    {
        qp = query_points[i_qp];
        merge_list_right = h_result_right + (k * i_qp);
        for (i_k = 0; i_k < k; ++i_k)
        {
            merge_list_left[i_k] = h_result[i_qp * k + i_k];
            h_result_right[i_qp * k + i_k] += (root + 1);
        }
        i_k = i_r = i_l = 0;
        for (i_k = 0; i_k < k; ++i_k)
        {
            if (cuDist(qp, tree[merge_list_left[i_l]]) < cuDist(qp, tree[merge_list_right[i_r]]))
            {
                h_result[i_qp * k + i_k] = merge_list_left[i_l];
                i_l++;
            }
            else
            {
                h_result[i_qp * k + i_k] = merge_list_right[i_r];
                i_r++;
            }
        }
        for (i_k = 0; i_k < k; ++i_k)
        {
            if (cuDist(qp, tree[root]) < cuDist(qp, tree[h_result[i_qp * k + i_k]]))
            {
                h_result[i_qp * k + i_k] = root;
                break;
            }
        }
    }
    free(merge_list_left);
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




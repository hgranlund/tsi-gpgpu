#include <kd-tree-naive.cuh>
#include <search-iterative.cuh>
#include <knn_gpgpu.h>
#include <point.h>

#include <stdio.h>
#include <helper_cuda.h>
#include "gtest/gtest.h"

bool isExpectedPoint(struct Point *tree, int n, float qx, float qy, float qz, float ex, float ey, float ez)
{
    // float dists[n];

    float query_point[3];
    query_point[0] = qx, query_point[1] = qy, query_point[2] = qz;
    
    // // int best_fit = nn(query_point, tree, dists, 0, midpoint(0, n));
    int mid = (int) floor((n) / 2);
    int best_fit = query_k(query_point, tree, 0, mid);

    float actual = tree[best_fit].p[0] + tree[best_fit].p[1] + tree[best_fit].p[2];
    float expected = ex + ey + ez;

    if (actual == expected)
    {
        return true;
    }
    return false;
}

TEST(search_iterative, search_iterative_wiki_correctness){
    int wn = 6;
    struct Point *wiki = (Point*) malloc(wn  * sizeof(Point));


    // (2,3), (5,4), (9,6), (4,7), (8,1), (7,2).
    wiki[0].p[0] = 2, wiki[0].p[1] = 3, wiki[0].p[2] = 0;
    wiki[1].p[0] = 5, wiki[1].p[1] = 4, wiki[1].p[2] = 0;
    wiki[2].p[0] = 9, wiki[2].p[1] = 6, wiki[2].p[2] = 0;
    wiki[3].p[0] = 4, wiki[3].p[1] = 7, wiki[3].p[2] = 0;
    wiki[4].p[0] = 8, wiki[4].p[1] = 1, wiki[4].p[2] = 0;
    wiki[5].p[0] = 7, wiki[5].p[1] = 2, wiki[5].p[2] = 0;

    cudaDeviceReset();
    build_kd_tree(wiki, wn);
    cashe_indexes(wiki, 0, wn, wn);
    
    ASSERT_EQ(true, isExpectedPoint(wiki, wn, 2, 3, 0, 2, 3, 0));
    ASSERT_EQ(true, isExpectedPoint(wiki, wn, 5, 4, 0, 5, 4, 0));
    ASSERT_EQ(true, isExpectedPoint(wiki, wn, 9, 6, 0, 9, 6, 0));
    ASSERT_EQ(true, isExpectedPoint(wiki, wn, 4, 7, 0, 4, 7, 0));
    ASSERT_EQ(true, isExpectedPoint(wiki, wn, 8, 1, 0, 8, 1, 0));
    ASSERT_EQ(true, isExpectedPoint(wiki, wn, 7, 2, 0, 7, 2, 0));
    ASSERT_EQ(true, isExpectedPoint(wiki, wn, 10, 10, 0, 9, 6, 0));
    ASSERT_EQ(true, isExpectedPoint(wiki, wn, 0, 0, 0, 2, 3, 0));
    ASSERT_EQ(true, isExpectedPoint(wiki, wn, 4, 4, 0, 5, 4, 0));
    ASSERT_EQ(true, isExpectedPoint(wiki, wn, 3, 2, 0, 2, 3, 0));
    ASSERT_EQ(true, isExpectedPoint(wiki, wn, 2, 6, 0, 4, 7, 0));
    ASSERT_EQ(true, isExpectedPoint(wiki, wn, 10, 0, 0, 8, 1, 0));
    ASSERT_EQ(true, isExpectedPoint(wiki, wn, 0, 10, 0, 4, 7, 0));
}

TEST(search_iterative, search_iterative_dfs){
    int wn = 6;
    struct Point *wiki = (Point*) malloc(wn  * sizeof(Point));


    // (2,3), (5,4), (9,6), (4,7), (8,1), (7,2).
    wiki[0].p[0] = 2, wiki[0].p[1] = 3, wiki[0].p[2] = 0;
    wiki[1].p[0] = 5, wiki[1].p[1] = 4, wiki[1].p[2] = 0;
    wiki[2].p[0] = 9, wiki[2].p[1] = 6, wiki[2].p[2] = 0;
    wiki[3].p[0] = 4, wiki[3].p[1] = 7, wiki[3].p[2] = 0;
    wiki[4].p[0] = 8, wiki[4].p[1] = 1, wiki[4].p[2] = 0;
    wiki[5].p[0] = 7, wiki[5].p[1] = 2, wiki[5].p[2] = 0;

    cudaDeviceReset();
    build_kd_tree(wiki, wn);
    cashe_indexes(wiki, 0, wn, wn);
    
    dfs(wiki, wn);
}

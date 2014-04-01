#include <kd-tree-naive.cuh>
#include <search-iterative.cuh>
#include <knn_gpgpu.h>
#include <point.h>

#include <stdio.h>
#include <helper_cuda.h>
#include "gtest/gtest.h"

#define debug 0

void _swap(struct Point *points, int a, int b)
{
    struct Point t = points[a];
    points[a] = points[b], points[b] = t;
}

int _midpoint(int lower, int upper)
{
    return (int) floor((float)(upper - lower) / 2) + lower;
}

float _quick_select(int k, struct Point *x, int lower, int upper, int dim)
{
    int pos, i,
        left = lower,
        right = upper - 1;

    float pivot;

    while (left < right)
    {
        pivot = x[k].p[dim];
        _swap(x, k, right);
        for (i = pos = left; i < right; i++)
        {
            if (x[i].p[dim] < pivot)
            {
                _swap(x, i, pos);
                pos++;
            }
        }
        _swap(x, right, pos);
        if (pos == k) break;
        if (pos < k) left = pos + 1;
        else right = pos - 1;
    }
    return x[k].p[dim];
}

void _center_median(struct Point *x, int lower, int upper, int dim)
{
    int i, r = _midpoint(lower, upper);

    float median = _quick_select(r, x, lower, upper, dim);

    for (i = lower; i < upper; ++i)
    {
        if (x[i].p[dim] == median)
        {
            _swap(x, i, r);
        }
    }
}

void _balance_branch(struct Point *x, int lower, int upper, int dim)
{
    if (lower >= upper) return;

    int i, r = _midpoint(lower, upper);

    _center_median(x, lower, upper, dim);

    upper--;

    for (i = lower; i < r; ++i)
    {
        if (x[i].p[dim] > x[r].p[dim])
        {
            while (x[upper].p[dim] > x[r].p[dim])
            {
                upper--;
            }
            _swap(x, i, upper);
        }
    }

    // To enable direct recursive execution.
    // _balance_branch(x, lower, r, 0);
    // _balance_branch(x, r + 1, upper, 0);
}

void _build_kd_tree(struct Point *x, int n)
{
    int i, j, p, step,
        h = ceil(log2((float) (n + 1)) - 1);
    for (i = 0; i < h; ++i)
    {
        p = pow(2.0, i);
        step = (int) floor((float)n / p);

        for (j = 0; j < p; ++j)
        {
            _balance_branch(x, (1 + step) * j, step * (1 + j), i % 3);
        }
    }
    return;
}

void print_t(Point *tree, int level, int lower, int upper, int n)
{
    if (debug)
    {
        if (lower >= upper)
        {
            return;
        }

        int i, r = floor((float)(upper - lower) / 2) + lower;

        printf("|");
        for (i = 0; i < level; ++i)
        {
            printf("--");
        }
        printf("(%3.1f, %3.1f, %3.1f)\n", tree[r].p[0], tree[r].p[1], tree[r].p[2]);

        print_t(tree, 1 + level, lower, r, n);
        print_t(tree, 1 + level, r + 1, upper, n);
    }
}

void _printPointsArray(Point *l, int n, char *s)
{
    if (debug)
    {
        printf("%10s: [ ", s);
        for (int i = 0; i < n; ++i)
        {
            printf("%3.1f, ", l[i].p[0]);
        }
        printf("]\n");
    }
}

bool isExpectedPoint(struct Point *tree, int n, float qx, float qy, float qz, float ex, float ey, float ez)
{
    // float dists[n];

    float query_point[3];
    query_point[0] = qx, query_point[1] = qy, query_point[2] = qz;

    // // int best_fit = nn(query_point, tree, dists, 0, _midpoint(0, n));
    int mid = (int) floor((float)(n) / 2);
    int best_fit = query_k(query_point, tree, 0, mid);

    float actual = tree[best_fit].p[0] + tree[best_fit].p[1] + tree[best_fit].p[2];
    float expected = ex + ey + ez;

    if (actual == expected)
    {
        return true;
    }
    return false;
}

TEST(search_iterative, search_iterative_wiki_correctness)
{
    int wn = 6;
    struct PointS *wiki = (PointS *) malloc(wn  * sizeof(PointS));
    struct Point *wiki_out = (Point *) malloc(wn  * sizeof(Point));


    // (2,3), (5,4), (9,6), (4,7), (8,1), (7,2).
    wiki[0].p[0] = 2, wiki[0].p[1] = 3, wiki[0].p[2] = 0;
    wiki[1].p[0] = 5, wiki[1].p[1] = 4, wiki[1].p[2] = 0;
    wiki[2].p[0] = 9, wiki[2].p[1] = 6, wiki[2].p[2] = 0;
    wiki[3].p[0] = 4, wiki[3].p[1] = 7, wiki[3].p[2] = 0;
    wiki[4].p[0] = 8, wiki[4].p[1] = 1, wiki[4].p[2] = 0;
    wiki[5].p[0] = 7, wiki[5].p[1] = 2, wiki[5].p[2] = 0;

    cudaDeviceReset();
    build_kd_tree(wiki, wn, wiki_out);
    cashe_indexes(wiki_out, 0, wn, wn);

    ASSERT_EQ(true, isExpectedPoint(wiki_out, wn, 2, 3, 0, 2, 3, 0));
    ASSERT_EQ(true, isExpectedPoint(wiki_out, wn, 5, 4, 0, 5, 4, 0));
    ASSERT_EQ(true, isExpectedPoint(wiki_out, wn, 9, 6, 0, 9, 6, 0));
    ASSERT_EQ(true, isExpectedPoint(wiki_out, wn, 4, 7, 0, 4, 7, 0));
    ASSERT_EQ(true, isExpectedPoint(wiki_out, wn, 8, 1, 0, 8, 1, 0));
    ASSERT_EQ(true, isExpectedPoint(wiki_out, wn, 7, 2, 0, 7, 2, 0));
    ASSERT_EQ(true, isExpectedPoint(wiki_out, wn, 10, 10, 0, 9, 6, 0));
    ASSERT_EQ(true, isExpectedPoint(wiki_out, wn, 0, 0, 0, 2, 3, 0));
    ASSERT_EQ(true, isExpectedPoint(wiki_out, wn, 4, 4, 0, 5, 4, 0));
    ASSERT_EQ(true, isExpectedPoint(wiki_out, wn, 3, 2, 0, 2, 3, 0));
    ASSERT_EQ(true, isExpectedPoint(wiki_out, wn, 2, 6, 0, 4, 7, 0));
    ASSERT_EQ(true, isExpectedPoint(wiki_out, wn, 10, 0, 0, 8, 1, 0));
    ASSERT_EQ(true, isExpectedPoint(wiki_out, wn, 0, 10, 0, 4, 7, 0));

    free(wiki);
    free(wiki_out);
}

TEST(search_iterative, search_iterative_dfs)
{
    int wn = 6;
    struct Point *wiki = (Point *) malloc(wn  * sizeof(Point));

    // (2,3), (5,4), (9,6), (4,7), (8,1), (7,2).
    wiki[0].p[0] = 2, wiki[0].p[1] = 3, wiki[0].p[2] = 0;
    wiki[1].p[0] = 5, wiki[1].p[1] = 4, wiki[1].p[2] = 0;
    wiki[2].p[0] = 9, wiki[2].p[1] = 6, wiki[2].p[2] = 0;
    wiki[3].p[0] = 4, wiki[3].p[1] = 7, wiki[3].p[2] = 0;
    wiki[4].p[0] = 8, wiki[4].p[1] = 1, wiki[4].p[2] = 0;
    wiki[5].p[0] = 7, wiki[5].p[1] = 2, wiki[5].p[2] = 0;

    // cudaDeviceReset();
    _build_kd_tree(wiki, wn);
    if (debug)
    {
        print_t(wiki, 0, 0, wn, wn);
        printf("\n");
    }

    cashe_indexes(wiki, 0, wn, wn);

    dfs(wiki, wn);
    free(wiki);
}

TEST(search_iterative, search_iterative_push)
{
    int eos = -1,
        *stack = (int *) malloc(5 * sizeof stack);

    push(stack, &eos, 3);
    ASSERT_EQ(3, stack[eos]);
    ASSERT_EQ(0, eos);

    push(stack, &eos, 42);
    ASSERT_EQ(42, stack[eos]);
    ASSERT_EQ(1, eos);

    push(stack, &eos, 1337);
    push(stack, &eos, 70012);
    push(stack, &eos, 1704);
    ASSERT_EQ(1704, stack[eos]);
    ASSERT_EQ(4, eos);

    free(stack);
}

TEST(search_iterative, search_iterative_pop)
{
    int eos = -1,
        *stack = (int *) malloc(5 * sizeof stack);

    ASSERT_EQ(-1, pop(stack, &eos));

    push(stack, &eos, 3);
    push(stack, &eos, 42);
    push(stack, &eos, 1337);

    ASSERT_EQ(1337, pop(stack, &eos));
    ASSERT_EQ(1, eos);
    ASSERT_EQ(42, pop(stack, &eos));
    ASSERT_EQ(0, eos);
    ASSERT_EQ(3, pop(stack, &eos));
    ASSERT_EQ(-1, eos);
    ASSERT_EQ(-1, pop(stack, &eos));
    ASSERT_EQ(-1, eos);

    free(stack);
}

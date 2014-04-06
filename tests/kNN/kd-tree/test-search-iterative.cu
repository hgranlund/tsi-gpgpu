#include <kd-tree-naive.cuh>
#include <search-iterative.cuh>
#include <knn_gpgpu.h>
#include <point.h>

#include <stdio.h>
#include <helper_cuda.h>
#include "gtest/gtest.h"



int store_locations_(Point *tree, int lower, int upper, int n)
{
    int r;

    if (lower >= upper)
    {
        return -1;
    }

    r = (int) ((upper - lower) / 2) + lower;

    tree[r].left = store_locations_(tree, lower, r, n);
    tree[r].right = store_locations_(tree, r + 1, upper, n);

    return r;
}


void _swap(struct Point *points, int a, int b)
{
    struct Point t = points[a];
    points[a] = points[b], points[b] = t;
}

int _midpoint(int lower, int upper)
{
    return (int) ((upper - lower) / 2) + lower;
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
        p = (int) pow(2.0, i);
        step = (int) (n / p);

        for (j = 0; j < p; ++j)
        {
            _balance_branch(x, (1 + step) * j, step * (1 + j), i % 3);
        }
    }
    return;
}

void print_t(Point *tree, int level, int lower, int upper, int n)
{
    if (lower >= upper)
    {
        return;
    }

    int i, r = ((upper - lower) / 2) + lower;

    printf("|");
    for (i = 0; i < level; ++i)
    {
        printf("--");
    }
    printf("(%3.1f, %3.1f, %3.1f)\n", tree[r].p[0], tree[r].p[1], tree[r].p[2]);

    print_t(tree, 1 + level, lower, r, n);
    print_t(tree, 1 + level, r + 1, upper, n);
}

void _printPointsArray(Point *l, int n, char *s)
{
    printf("%10s: [ ", s);
    for (int i = 0; i < n; ++i)
    {
        printf("%3.1f, ", l[i].p[0]);
    }
    printf("]\n");
}

bool isExpectedPoint(struct Point *tree, int n, float qx, float qy, float qz, float ex, float ey, float ez)
{
    // float dists[n];

    float query_point[3];
    query_point[0] = qx, query_point[1] = qy, query_point[2] = qz;

    // // int best_fit = nn(query_point, tree, dists, 0, _midpoint(0, n));
    int best_fit = query_a(query_point, tree, n);

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

// TEST(kd_search, kd_search_timing_cpu)
// {
//     int i, n = 1000000;
//     PointS *points;
//     Point *points_out;
//     points = (PointS *) malloc(n  * sizeof(PointS));
//     points_out = (Point *) malloc(n  * sizeof(Point));
//     srand(time(NULL));

//     for (i = 0; i < n; ++i)
//     {
//         PointS t;
//         t.p[0] = rand();
//         t.p[1] = rand();
//         t.p[2] = rand();
//         points[i] = t;
//     }

//     cudaDeviceReset();



//     build_kd_tree(points, n, points_out);


//     store_locations_(points_out, 0, n, n);

//     int test_runs = 1;
//     float **query_data = (float **) malloc(test_runs * sizeof * query_data);

//     for (i = 0; i < test_runs; i++)
//     {
//         query_data[i] = (float *) malloc(3 * sizeof * query_data[i]);
//         query_data[i][0] = rand() % 1000;
//         query_data[i][1] = rand() % 1000;
//         query_data[i][2] = rand() % 1000;
//     }

//     cudaEvent_t start, stop;
//     unsigned int bytes = n * (sizeof(Point) + sizeof(int));
//     checkCudaErrors(cudaEventCreate(&start));
//     checkCudaErrors(cudaEventCreate(&stop));
//     float elapsed_time = 0;

//     checkCudaErrors(cudaEventRecord(start, 0));

//     for (i = 0; i < test_runs; i++)
//     {
//         query_a(query_data[i], points_out,  n);
//     }


//     checkCudaErrors(cudaEventRecord(stop, 0));
//     cudaEventSynchronize(start);
//     cudaEventSynchronize(stop);
//     cudaEventElapsedTime(&elapsed_time, start, stop);
//     elapsed_time = elapsed_time ;
//     double throughput = 1.0e-9 * ((double)bytes) / (elapsed_time * 1e-3);
//     printf("query_a, Throughput = %.4f GB/s, Time = %.5f ms, Size = %u Elements, NumDevsUsed = %d\n",
//            throughput, elapsed_time, n, 1);


//     for (i = 0; i < test_runs; ++i)
//     {
//         free(query_data[i]);
//     }
//     free(query_data);

//     free(points);
// }

// TEST(search_iterative, search_iterative_dfs)
// {
//     int wn = 6;
//     struct Point *wiki = (Point *) malloc(wn  * sizeof(Point));

//     // (2,3), (5,4), (9,6), (4,7), (8,1), (7,2).
//     wiki[0].p[0] = 2, wiki[0].p[1] = 3, wiki[0].p[2] = 0;
//     wiki[1].p[0] = 5, wiki[1].p[1] = 4, wiki[1].p[2] = 0;
//     wiki[2].p[0] = 9, wiki[2].p[1] = 6, wiki[2].p[2] = 0;
//     wiki[3].p[0] = 4, wiki[3].p[1] = 7, wiki[3].p[2] = 0;
//     wiki[4].p[0] = 8, wiki[4].p[1] = 1, wiki[4].p[2] = 0;
//     wiki[5].p[0] = 7, wiki[5].p[1] = 2, wiki[5].p[2] = 0;

//     // cudaDeviceReset();
//     _build_kd_tree(wiki, wn);
//     print_t(wiki, 0, 0, wn, wn);
//     printf("\n");

//     cashe_indexes(wiki, 0, wn, wn);

//     dfs(wiki, wn);
// }

// TEST(search_iterative, search_iterative_query_a)
// {
//     int wn = 6;
//     struct Point *wiki = (Point *) malloc(wn  * sizeof(Point)),
//                   *qp = (Point *) malloc(sizeof(Point));

//     // (2,3), (5,4), (9,6), (4,7), (8,1), (7,2).
//     wiki[0].p[0] = 2, wiki[0].p[1] = 3, wiki[0].p[2] = 0;
//     wiki[1].p[0] = 5, wiki[1].p[1] = 4, wiki[1].p[2] = 0;
//     wiki[2].p[0] = 9, wiki[2].p[1] = 6, wiki[2].p[2] = 0;
//     wiki[3].p[0] = 4, wiki[3].p[1] = 7, wiki[3].p[2] = 0;
//     wiki[4].p[0] = 8, wiki[4].p[1] = 1, wiki[4].p[2] = 0;
//     wiki[5].p[0] = 7, wiki[5].p[1] = 2, wiki[5].p[2] = 0;

//     // cudaDeviceReset();
//     _build_kd_tree(wiki, wn);
//     print_t(wiki, 0, 0, wn, wn);
//     printf("\n");

//     cashe_indexes(wiki, 0, wn, wn);

//     query_a(qp, wiki, wn);
// }

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

TEST(search_iterative, search_iterative_peek)
{
    int eos = -1,
        *stack = (int *) malloc(5 * sizeof stack);

    ASSERT_EQ(-1, peek(stack, eos));

    push(stack, &eos, 1337);

    ASSERT_EQ(1337, peek(stack, eos));
    ASSERT_EQ(1337, peek(stack, eos));
    ASSERT_EQ(1337, peek(stack, eos));

    free(stack);
}

TEST(search_iterative, search_iterative_find)
{
    int eos = -1,
        *stack = (int *) malloc(5 * sizeof stack);

    ASSERT_EQ(-1, find(stack, eos, -1));

    push(stack, &eos, 1337);

    ASSERT_EQ(0, find(stack, eos, 1337));

    push(stack, &eos, 3);
    push(stack, &eos, 42);

    ASSERT_EQ(1, find(stack, eos, 3));
    ASSERT_EQ(2, find(stack, eos, 42));
    ASSERT_EQ(-1, find(stack, eos, 1704));

    free(stack);
}

TEST(search_iterative, search_iterative_upDim)
{
    int dim = 0;

    upDim(&dim);
    ASSERT_EQ(1, dim);

    upDim(&dim);
    ASSERT_EQ(2, dim);

    upDim(&dim);
    ASSERT_EQ(0, dim);

    upDim(&dim);
    ASSERT_EQ(1, dim);
}

TEST(search_iterative, search_iterative_downDim)
{
    int dim = 0;

    downDim(&dim);
    ASSERT_EQ(2, dim);

    downDim(&dim);
    ASSERT_EQ(1, dim);

    downDim(&dim);
    ASSERT_EQ(0, dim);

    downDim(&dim);
    ASSERT_EQ(2, dim);
}

#include <float.h>
#include <math.h>

#include "test-common.cuh"
#include "kd-search-openmp.cuh"
#include <knn_gpgpu.h>

bool isExpectedPoint_(struct Node *tree, int n, int k,  float qx, float qy, float qz, float ex, float ey, float ez)
{
    struct Point query_point;

    struct SPoint *s_stack_ptr = (struct SPoint *)malloc(51 * sizeof(struct SPoint));
    struct KPoint *k_stack_ptr = (struct KPoint *) malloc((k + 1) * sizeof(struct KPoint));

    int *result = (int *) malloc(k * sizeof(int));

    query_point.p[0] = qx, query_point.p[1] = qy, query_point.p[2] = qz;

    kNN(query_point, tree, n, k, result, s_stack_ptr, k_stack_ptr);

    float actual = tree[result[0]].p[0] + tree[result[0]].p[1] + tree[result[0]].p[2];
    float expected = ex + ey + ez;

    // printf(">> WP tree\nsearching for (%3.1f, %3.1f, %3.1f)\n"
    //        "found (%3.1f, %3.1f, %3.1f) seen %d nodes\n\n",
    //        qx, qy, qz,
    //        tree[result[0]].p[0], tree[result[0]].p[1], tree[result[0]].p[2], visited);

    free(s_stack_ptr);
    free(k_stack_ptr);
    free(result);

    if (actual == expected)
    {
        return true;
    }
    return false;
}

TEST(kd_search_openmp, isEmpty)
{
    struct SPoint *stack_ptr = (struct SPoint *) malloc(4 * sizeof(struct SPoint)),
                   *stack = stack_ptr,
                    value1;

    initStack(&stack);

    value1.index = 10;

    ASSERT_TRUE(isEmpty(stack));

    stack[0] = value1;
    stack++;
    ASSERT_FALSE(isEmpty(stack));

    free(stack_ptr);
}

TEST(kd_search_openmp, push)
{
    struct SPoint *stack_ptr = (struct SPoint *) malloc(3 * sizeof(struct SPoint)),
                   *stack = stack_ptr,
                    value1,
                    value2;

    initStack(&stack);

    value1.index = 1;
    value2.index = 3;

    push(&stack, value1);
    push(&stack, value2);
    ASSERT_EQ(value1.index, stack_ptr[1].index);
    ASSERT_EQ(value2.index, stack_ptr[2].index);

    free(stack_ptr);
}

TEST(kd_search_openmp, pop)
{
    struct SPoint *stack_ptr = (struct SPoint *) malloc(4 * sizeof(struct SPoint)),
                   *stack = stack_ptr,
                    value1,
                    value2,
                    value3;

    initStack(&stack);

    value1.index = 1;
    value2.index = 2;
    value3.index = 3;

    stack[0] = value1;
    stack[1] = value2;
    stack[2] = value3;
    stack += 3;

    ASSERT_EQ(value3.index, pop(&stack).index);
    ASSERT_EQ(value2.index, pop(&stack).index);
    ASSERT_EQ(value1.index, pop(&stack).index);

    free(stack_ptr);
}

TEST(kd_search_openmp, peek)
{
    struct SPoint *stack_ptr = (struct SPoint *) malloc(4 * sizeof(struct SPoint)),
                   *stack = stack_ptr,
                    value1;

    initStack(&stack);

    value1.index = 10;

    ASSERT_EQ(-1, peek(stack).index);
    ASSERT_EQ(-1, peek(stack).index);

    push(&stack, value1);

    ASSERT_EQ(value1.index, peek(stack).index);
    ASSERT_EQ(value1.index, peek(stack).index);

    free(stack_ptr);
}

TEST(kd_search_openmp, initKStack)
{
    struct KPoint *k_stack_ptr = (struct KPoint *) malloc(51 * sizeof(struct KPoint)),
                   *k_stack = k_stack_ptr;

    initKStack(&k_stack, 50);

    ASSERT_EQ(-1, k_stack[-1].dist);
    ASSERT_EQ(FLT_MAX, k_stack[0].dist);
    ASSERT_EQ(FLT_MAX, k_stack[49].dist);

    free(k_stack_ptr);
}

TEST(kd_search_openmp, insert)
{
    int n = 3;
    struct KPoint *k_stack_ptr = (struct KPoint *) malloc(51 * sizeof(struct KPoint)),
                   *k_stack = k_stack_ptr;

    initKStack(&k_stack, n);
    struct KPoint a, b, c, d;

    a.dist = 1;
    b.dist = 2;
    c.dist = 3;
    d.dist = 0;

    insert(k_stack, a, n);
    ASSERT_EQ(FLT_MAX, look(k_stack, n).dist);
    ASSERT_EQ(a.dist, k_stack[0].dist);

    insert(k_stack, b, n);
    ASSERT_EQ(FLT_MAX, look(k_stack, n).dist);
    ASSERT_EQ(b.dist, k_stack[1].dist);

    insert(k_stack, c, n);
    ASSERT_EQ(c.dist, look(k_stack, n).dist);
    ASSERT_EQ(c.dist, k_stack[2].dist);

    insert(k_stack, d, n);
    ASSERT_EQ(b.dist, look(k_stack, n).dist);
    ASSERT_EQ(d.dist, k_stack[0].dist);

    free(k_stack_ptr);
}

TEST(kd_search_openmp, insert_k_is_one)
{
    int n = 1;
    struct KPoint *k_stack_ptr = (struct KPoint *) malloc(51 * sizeof(struct KPoint)),
                   *k_stack = k_stack_ptr;

    initKStack(&k_stack, n);
    struct KPoint a, b;

    a.dist = 1;
    b.dist = 0;

    insert(k_stack, a, n);
    ASSERT_EQ(a.dist, look(k_stack, n).dist);
    ASSERT_EQ(a.dist, k_stack[0].dist);

    insert(k_stack, b, n);
    ASSERT_EQ(b.dist, look(k_stack, n).dist);
    ASSERT_EQ(b.dist, k_stack[0].dist);

    free(k_stack_ptr);
}

TEST(kd_search_openmp, upDim)
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

TEST(kd_search_openmp, correctness_with_k)
{
    int n = 6,
        k = 3;

    int *result = (int *) malloc(k * sizeof(int));

    struct Point *points = (struct Point *) malloc(n  * sizeof(struct Point));
    struct Node *tree = (struct Node *) malloc(n  * sizeof(struct Node));

    struct SPoint *s_stack_ptr = (struct SPoint *)malloc(51 * sizeof(struct SPoint));
    struct KPoint *k_stack_ptr = (struct KPoint *) malloc((k + 1) * sizeof(struct KPoint));

    points[0].p[0] = 2, points[0].p[1] = 3, points[0].p[2] = 0;
    points[1].p[0] = 5, points[1].p[1] = 4, points[1].p[2] = 0;
    points[2].p[0] = 9, points[2].p[1] = 6, points[2].p[2] = 0;
    points[3].p[0] = 4, points[3].p[1] = 7, points[3].p[2] = 0;
    points[4].p[0] = 8, points[4].p[1] = 1, points[4].p[2] = 0;
    points[5].p[0] = 7, points[5].p[1] = 2, points[5].p[2] = 0;

    cudaDeviceReset();
    build_kd_tree(points, n, tree);

    cudaDeviceReset();
    kNN(points[4], tree, n, k, result, s_stack_ptr, k_stack_ptr);

    ASSERT_EQ(4, result[0]);
    ASSERT_EQ(3, result[1]);
    ASSERT_EQ(1, result[2]);

    free(points);
    free(tree);

    free(s_stack_ptr);
    free(k_stack_ptr);
}

TEST(kd_search_openmp, correctness_with_10000_points_file)
{
    int n, k = 1;

    for (n = 1000; n <= 10000; n += 1000)
    {
        struct Point *points = (struct Point *) malloc(n  * sizeof(struct Point));
        struct Node *tree = (struct Node *) malloc(n  * sizeof(struct Node));

        srand(time(NULL));

        readPoints("../tests/data/10000_points.data", n, points);

        cudaDeviceReset();
        build_kd_tree(points, n, tree);

        // printTree(tree, 0, n / 2);

        int *result = (int *) malloc(k * sizeof(int));

        int i,
            test_runs = n;

        struct SPoint *stack_ptr = (struct SPoint *)malloc(51 * sizeof(struct SPoint));
        struct KPoint *k_stack_ptr = (struct KPoint *) malloc((k + 1) * sizeof(struct KPoint));

        for (i = 0; i < test_runs; ++i)
        {
            cudaDeviceReset();
            kNN(points[i], tree, n, k, result, stack_ptr, k_stack_ptr);

            // printf("Looking for (%3.1f, %3.1f, %3.1f), found (%3.1f, %3.1f, %3.1f)\n",
            //        tree[i].p[0], tree[i].p[1], tree[i].p[2],
            //        tree[result[0]].p[0], tree[result[0]].p[1], tree[result[0]].p[2]);

            ASSERT_EQ(points[i].p[0], tree[result[0]].p[0]) << "Failed at i = " << i << " with n = " << n ;
            ASSERT_EQ(points[i].p[1], tree[result[0]].p[1]) << "Failed at i = " << i << " with n = " << n;
            ASSERT_EQ(points[i].p[2], tree[result[0]].p[2]) << "Failed at i = " << i << " with n = " << n;
        }

        free(tree);
        free(result);
        free(points);
        free(stack_ptr);
        free(k_stack_ptr);
    };
};

// TEST(kd_search_openmp, queryAll_correctness_with_10000_points_file)
// {
//     int n, i, k = 1;

//     for (n = 1000; n <= 10000; n += 1000)
//     {
//         struct Point *points = (struct Point *) malloc(n  * sizeof(struct Point));
//         struct Node *tree = (struct Node *) malloc(n  * sizeof(struct Node));

//         srand(time(NULL));

//         readPoints("../tests/data/10000_points.data", n, points);

//         cudaDeviceReset();
//         build_kd_tree(points, n, tree);

//         // printTree(tree, 0, n / 2);

//         int *result = (int *) malloc(n * k * sizeof(int));


//         queryAll(points, tree, n, n, 1, result);
//         for (i = 0; i < n; ++i)
//         {
//             ASSERT_EQ(points[i].p[0], tree[result[i]].p[0]) << "Failed at i = " << i << " with n = " << n ;
//             ASSERT_EQ(points[i].p[1], tree[result[i]].p[1]) << "Failed at i = " << i << " with n = " << n;
//             ASSERT_EQ(points[i].p[2], tree[result[i]].p[2]) << "Failed at i = " << i << " with n = " << n;
//         }

//         free(tree);
//         free(result);
//         free(points);
//     };
// };

TEST(kd_search_openmp, knn_wikipedia_example)
{
    int n = 6,
        k = 1;

    struct Point *points = (struct Point *) malloc(n  * sizeof(struct Point));
    struct Node *tree = (struct Node *) malloc(n  * sizeof(struct Node));

    points[0].p[0] = 2, points[0].p[1] = 3, points[0].p[2] = 0;
    points[1].p[0] = 5, points[1].p[1] = 4, points[1].p[2] = 0;
    points[2].p[0] = 9, points[2].p[1] = 6, points[2].p[2] = 0;
    points[3].p[0] = 4, points[3].p[1] = 7, points[3].p[2] = 0;
    points[4].p[0] = 8, points[4].p[1] = 1, points[4].p[2] = 0;
    points[5].p[0] = 7, points[5].p[1] = 2, points[5].p[2] = 0;

    cudaDeviceReset();
    build_kd_tree(points, n, tree);


    ASSERT_EQ(true, isExpectedPoint_(tree, n, k, 2, 3, 0, 2, 3, 0));
    ASSERT_EQ(true, isExpectedPoint_(tree, n, k, 5, 4, 0, 5, 4, 0));
    ASSERT_EQ(true, isExpectedPoint_(tree, n, k, 9, 6, 0, 9, 6, 0));
    ASSERT_EQ(true, isExpectedPoint_(tree, n, k, 4, 7, 0, 4, 7, 0));
    ASSERT_EQ(true, isExpectedPoint_(tree, n, k, 8, 1, 0, 8, 1, 0));
    ASSERT_EQ(true, isExpectedPoint_(tree, n, k, 7, 2, 0, 7, 2, 0));
    ASSERT_EQ(true, isExpectedPoint_(tree, n, k, 10, 10, 0, 9, 6, 0));
    ASSERT_EQ(true, isExpectedPoint_(tree, n, k, 0, 0, 0, 2, 3, 0));
    ASSERT_EQ(true, isExpectedPoint_(tree, n, k, 4, 4, 0, 5, 4, 0));
    ASSERT_EQ(true, isExpectedPoint_(tree, n, k, 3, 2, 0, 2, 3, 0));
    ASSERT_EQ(true, isExpectedPoint_(tree, n, k, 2, 6, 0, 4, 7, 0));
    ASSERT_EQ(true, isExpectedPoint_(tree, n, k, 10, 0, 0, 8, 1, 0));
    ASSERT_EQ(true, isExpectedPoint_(tree, n, k, 0, 10, 0, 4, 7, 0));

    free(points);
    free(tree);
}

TEST(kd_search_openmp, query_all_wikipedia_example)
{
    int n = 6, k = 1;
    struct Point *points = (struct Point *) malloc(n * sizeof(struct Point));
    struct Node *tree = (struct Node *) malloc(n * sizeof(struct Node));
    int *result = (int *) malloc(n * k * sizeof(int));

    points[0].p[0] = 2, points[0].p[1] = 3, points[0].p[2] = 0;
    points[1].p[0] = 5, points[1].p[1] = 4, points[1].p[2] = 0;
    points[5].p[0] = 7, points[5].p[1] = 2, points[5].p[2] = 0;

    cudaDeviceReset();
    build_kd_tree(points, n, tree);

    mpQueryAll(points, tree, n, n, 1, result);

    ASSERT_EQ(result[0], 0);
    ASSERT_EQ(result[1], 1);
    ASSERT_EQ(result[2], 5);
    ASSERT_EQ(result[3], 2);
    ASSERT_EQ(result[4], 4);
    ASSERT_EQ(result[5], 3);

    free(points);
    free(tree);
    free(result);
}

TEST(kd_search_openmp, knn_timing)
{
    int n, k = 1;

    for (n = 10000; n <= 10000; n += 1000)
    {
        struct Point *points = (struct Point *) malloc(n  * sizeof(struct Point));
        struct Node *tree = (struct Node *) malloc(n  * sizeof(struct Node));

        struct SPoint *stack_ptr = (struct SPoint *)malloc(51 * sizeof(struct SPoint));
        struct KPoint *k_stack_ptr = (struct KPoint *) malloc((k + 1) * sizeof(struct KPoint));

        int *result = (int *) malloc(k * sizeof(int));

        srand(time(NULL));

        readPoints("../tests/data/10000_points.data", n, points);

        cudaDeviceReset();
        build_kd_tree(points, n, tree);

        int i,
            test_runs = n;

        long start_time = startTiming();
        for (i = 0; i < test_runs; ++i)
        {
            kNN(points[i], tree, n, k, result, stack_ptr, k_stack_ptr);
        }
        printf("Time = %ld ms, Size = %d Elements\n", endTiming(start_time), n);

        free(tree);
        free(result);
        free(points);
        free(stack_ptr);
        free(k_stack_ptr);
    };
};

TEST(kd_search_openmp, query_all_timing)
{
    int n, k = 5;

    for (n = 10000; n <= 10000; n += 1000)
    {
        struct Point *points = (struct Point *) malloc(n  * sizeof(struct Point));
        struct Node *tree = (struct Node *) malloc(n  * sizeof(struct Node));

        readPoints("../tests/data/10000_points.data", n, points);

        cudaDeviceReset();
        build_kd_tree(points, n, tree);

        int test_runs = n;
        int *result = (int *) malloc(test_runs * k * sizeof(int));

        cudaEvent_t start, stop;
        float elapsed_time = 0;
        int bytes = n * (sizeof(struct Node));

        cudaStartTiming(start, stop, elapsed_time);
        queryAll(points, tree, test_runs, n, k, result);
        cudaStopTiming(start, stop, elapsed_time);
        printCudaTiming(elapsed_time, bytes, n);

        free(tree);
        free(points);
        cudaDeviceReset();
    };
};

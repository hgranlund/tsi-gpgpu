#include <search-iterative.cuh>
#include <knn_gpgpu.h>
#include <float.h>

#include "test-common.cuh"

bool isExpectedPoint(struct Point *tree, int n, int k,  float qx, float qy, float qz, float ex, float ey, float ez)
{
    struct Point query_point;
    int result[k];
    query_point.p[0] = qx, query_point.p[1] = qy, query_point.p[2] = qz;

    kNN(query_point, tree, n, k, result);
    float actual = tree[result[0]].p[0] + tree[result[0]].p[1] + tree[result[0]].p[2];
    float expected = ex + ey + ez;

    if (actual == expected)
    {
        return true;
    }
    return false;
}

TEST(search_iterative, wikipedia_example)
{
    int n = 6,
        k = 1;

    struct PointS *points = (struct PointS *) malloc(n  * sizeof(PointS));
    struct Point *points_out = (struct Point *) malloc(n  * sizeof(Point));

    points[0].p[0] = 2, points[0].p[1] = 3, points[0].p[2] = 0;
    points[1].p[0] = 5, points[1].p[1] = 4, points[1].p[2] = 0;
    points[2].p[0] = 9, points[2].p[1] = 6, points[2].p[2] = 0;
    points[3].p[0] = 4, points[3].p[1] = 7, points[3].p[2] = 0;
    points[4].p[0] = 8, points[4].p[1] = 1, points[4].p[2] = 0;
    points[5].p[0] = 7, points[5].p[1] = 2, points[5].p[2] = 0;

    cudaDeviceReset();
    build_kd_tree(points, n, points_out);


    ASSERT_EQ(true, isExpectedPoint(points_out, n, k, 2, 3, 0, 2, 3, 0));
    ASSERT_EQ(true, isExpectedPoint(points_out, n, k, 5, 4, 0, 5, 4, 0));
    ASSERT_EQ(true, isExpectedPoint(points_out, n, k, 9, 6, 0, 9, 6, 0));
    ASSERT_EQ(true, isExpectedPoint(points_out, n, k, 4, 7, 0, 4, 7, 0));
    ASSERT_EQ(true, isExpectedPoint(points_out, n, k, 8, 1, 0, 8, 1, 0));
    ASSERT_EQ(true, isExpectedPoint(points_out, n, k, 7, 2, 0, 7, 2, 0));
    ASSERT_EQ(true, isExpectedPoint(points_out, n, k, 10, 10, 0, 9, 6, 0));
    ASSERT_EQ(true, isExpectedPoint(points_out, n, k, 0, 0, 0, 2, 3, 0));
    ASSERT_EQ(true, isExpectedPoint(points_out, n, k, 4, 4, 0, 5, 4, 0));
    ASSERT_EQ(true, isExpectedPoint(points_out, n, k, 3, 2, 0, 2, 3, 0));
    ASSERT_EQ(true, isExpectedPoint(points_out, n, k, 2, 6, 0, 4, 7, 0));
    ASSERT_EQ(true, isExpectedPoint(points_out, n, k, 10, 0, 0, 8, 1, 0));
    ASSERT_EQ(true, isExpectedPoint(points_out, n, k, 0, 10, 0, 4, 7, 0));

}

TEST(search_iterative, correctness_with_k)
{
    int n = 6,
        k = 3,
        result[k];


    struct PointS *points = (struct PointS *) malloc(n  * sizeof(PointS));
    struct Point *points_out = (struct Point *) malloc(n  * sizeof(Point));

    points[0].p[0] = 2, points[0].p[1] = 3, points[0].p[2] = 0;
    points[1].p[0] = 5, points[1].p[1] = 4, points[1].p[2] = 0;
    points[2].p[0] = 9, points[2].p[1] = 6, points[2].p[2] = 0;
    points[3].p[0] = 4, points[3].p[1] = 7, points[3].p[2] = 0;
    points[4].p[0] = 8, points[4].p[1] = 1, points[4].p[2] = 0;
    points[5].p[0] = 7, points[5].p[1] = 2, points[5].p[2] = 0;

    cudaDeviceReset();
    build_kd_tree(points, n, points_out);
    kNN(points_out[4], points_out, n, k, result);

    ASSERT_EQ(4, result[0]);
    ASSERT_EQ(3, result[1]);
    ASSERT_EQ(1, result[2]);

}

TEST(search_iterative, push)
{
    int stack_init[50],
        *stack;

    initStack(stack_init, &stack);

    push(&stack, 1);
    push(&stack, 3);
    ASSERT_EQ(1, stack_init[1]);
    ASSERT_EQ(3, stack_init[2]);
}

TEST(search_iterative, pop)
{
    int stack_init[50],
        *stack;

    initStack(stack_init, &stack);

    stack_init[1] = 1;
    stack_init[2] = 2;
    stack_init[3] = 3;
    stack += 3;

    ASSERT_EQ(3, pop(&stack));
    ASSERT_EQ(2, pop(&stack));
    ASSERT_EQ(1, pop(&stack));
}

TEST(search_iterative, isEmpty)
{
    int stack_init[50],
        *stack;

    initStack(stack_init, &stack);
    ASSERT_TRUE(isEmpty(stack));

    stack_init[1] = 10;
    stack++;
    ASSERT_FALSE(isEmpty(stack));
}

TEST(search_iterative, peek)
{
    int stack_init[50],
        *stack;

    initStack(stack_init, &stack);

    ASSERT_EQ(-1, peek(stack));
    ASSERT_EQ(-1, peek(stack));

    stack_init[1] = -1;
    stack++;
    stack_init[2] = 10;
    stack++;

    ASSERT_EQ(10, peek(stack));
    ASSERT_EQ(10, peek(stack));
}

TEST(search_iterative, upDim)
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


TEST(search_iterative, initKStack)
{
    struct KPoint k_stack_init[51],
            *k_stack = k_stack_init;

    initKStack(&k_stack, 50);

    ASSERT_EQ(-1, k_stack[-1].dist);
    ASSERT_EQ(FLT_MAX, k_stack[0].dist);
    ASSERT_EQ(FLT_MAX, k_stack[49].dist);
}

TEST(search_iterative, insert)
{
    int n = 3;
    struct KPoint k_stack_init[n + 1],
            *k_stack = k_stack_init;

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
}

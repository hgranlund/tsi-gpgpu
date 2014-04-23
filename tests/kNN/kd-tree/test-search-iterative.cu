#include <search-iterative.cuh>
#include <knn_gpgpu.h>
#include <float.h>

#include "test-common.cuh"

bool isExpectedPoint(struct Point *tree, int n, float qx, float qy, float qz, float ex, float ey, float ez)
{
    struct Point query_point;
    query_point.p[0] = qx, query_point.p[1] = qy, query_point.p[2] = qz;

    int best_fit = query_a(query_point, tree, n);

    float actual = tree[best_fit].p[0] + tree[best_fit].p[1] + tree[best_fit].p[2];
    float expected = ex + ey + ez;

    if (actual == expected)
    {
        return true;
    }
    return false;
}

TEST(search_iterative, wikipedia_example)
{
    int n = 6;

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

    ASSERT_EQ(true, isExpectedPoint(points_out, n, 2, 3, 0, 2, 3, 0));
    ASSERT_EQ(true, isExpectedPoint(points_out, n, 5, 4, 0, 5, 4, 0));
    ASSERT_EQ(true, isExpectedPoint(points_out, n, 9, 6, 0, 9, 6, 0));
    ASSERT_EQ(true, isExpectedPoint(points_out, n, 4, 7, 0, 4, 7, 0));
    ASSERT_EQ(true, isExpectedPoint(points_out, n, 8, 1, 0, 8, 1, 0));
    ASSERT_EQ(true, isExpectedPoint(points_out, n, 7, 2, 0, 7, 2, 0));
    ASSERT_EQ(true, isExpectedPoint(points_out, n, 10, 10, 0, 9, 6, 0));
    ASSERT_EQ(true, isExpectedPoint(points_out, n, 0, 0, 0, 2, 3, 0));
    ASSERT_EQ(true, isExpectedPoint(points_out, n, 4, 4, 0, 5, 4, 0));
    ASSERT_EQ(true, isExpectedPoint(points_out, n, 3, 2, 0, 2, 3, 0));
    ASSERT_EQ(true, isExpectedPoint(points_out, n, 2, 6, 0, 4, 7, 0));
    ASSERT_EQ(true, isExpectedPoint(points_out, n, 10, 0, 0, 8, 1, 0));
    ASSERT_EQ(true, isExpectedPoint(points_out, n, 0, 10, 0, 4, 7, 0));
}

TEST(search_iterative, inorder_print)
{
    int n = 6;

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

    query_a(points_out[0], points_out, n);
}

TEST(search_iterative, push)
{
    int stack[50],
        *stackPtr;

    initStack(stack, &stackPtr);

    push(&stackPtr, 1);
    push(&stackPtr, 3);
    ASSERT_EQ(1, stack[1]);
    ASSERT_EQ(3, stack[2]);
}

TEST(search_iterative, pop)
{
    int stack[50],
        *stackPtr;

    initStack(stack, &stackPtr);


    stack[1] = 1;
    stack[2] = 2;
    stack[3] = 3;
    stackPtr += 3;
    ASSERT_EQ(3, pop(&stackPtr));
    ASSERT_EQ(2, pop(&stackPtr));
    ASSERT_EQ(1, pop(&stackPtr));
}

TEST(search_iterative, isEmpty)
{
    int stack[50],
        *stackPtr;

    initStack(stack, &stackPtr);

    ASSERT_TRUE(isEmpty(stackPtr));

    stack[1] = 10;
    stackPtr++;
    ASSERT_FALSE(isEmpty(stackPtr));

}
TEST(search_iterative, peek)
{
    int stack[50],
        *stackPtr;

    initStack(stack, &stackPtr);

    ASSERT_EQ(-1, peek(stackPtr));
    ASSERT_EQ(-1, peek(stackPtr));
    stack[1] = -1;
    stackPtr++;
    stack[2] = 10;
    stackPtr++;
    ASSERT_EQ(10, peek(stackPtr));
    ASSERT_EQ(10, peek(stackPtr));
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
    struct KPoint kStack[51],
            *kStackPtr = kStack;

    initKStack(&kStackPtr, 50);

    ASSERT_EQ(-1, kStackPtr[-1].dist);
    ASSERT_EQ(FLT_MAX, kStackPtr[0].dist);
    ASSERT_EQ(FLT_MAX, kStackPtr[49].dist);
}

TEST(search_iterative, insert)
{
    int n = 3;
    struct KPoint kStack[n + 1],
            *kStackPtr = kStack;

    initKStack(&kStackPtr, n);
    struct KPoint a, b, c, d;
    a.dist = 1;
    b.dist = 2;
    c.dist = 3;
    d.dist = 0;

    insert(kStackPtr, a, n);
    ASSERT_EQ(FLT_MAX, look(kStackPtr, n).dist);
    ASSERT_EQ(a.dist, kStackPtr[0].dist);

    insert(kStackPtr, b, n);
    ASSERT_EQ(FLT_MAX, look(kStackPtr, n).dist);
    ASSERT_EQ(b.dist, kStackPtr[1].dist);

    insert(kStackPtr, c, n);
    ASSERT_EQ(c.dist, look(kStackPtr, n).dist);
    ASSERT_EQ(c.dist, kStackPtr[2].dist);

    insert(kStackPtr, d, n);
    ASSERT_EQ(b.dist, look(kStackPtr, n).dist);
    ASSERT_EQ(d.dist, kStackPtr[0].dist);
}


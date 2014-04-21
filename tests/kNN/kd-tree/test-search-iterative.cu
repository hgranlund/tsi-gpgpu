#include <search-iterative.cuh>
#include <knn_gpgpu.h>
#include "test-common.cuh"

bool isExpectedPoint(struct Point *tree, int n, float qx, float qy, float qz, float ex, float ey, float ez)
{
    float query_point[3];
    query_point[0] = qx, query_point[1] = qy, query_point[2] = qz;

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

TEST(search_iterative, push)
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

TEST(search_iterative, pop)
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

TEST(search_iterative, peek)
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

TEST(search_iterative, find)
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

TEST(search_iterative, downDim)
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

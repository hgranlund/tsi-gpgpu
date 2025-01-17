#include <knn_gpgpu.h>
#include "test-common.cuh"

void cuPrintTree(struct Node *tree, int level, int root)
{
    if (root < 0) return;

    int i;

    printf("|");
    for (i = 0; i < level; ++i)
    {
        printf("----");
    }
    printf("(%3.1f, %3.1f, %3.1f): dim = %d \n", tree[root].p[0], tree[root].p[1], tree[root].p[2], level % 3);

    cuPrintTree(tree, 1 + level, tree[root].left);
    cuPrintTree(tree, 1 + level, tree[root].right);
}

TEST(kd_tree_build, correctness_diagonal)
{
    int i, n = 8;
    float temp;

    struct Point *points = (struct Point *) malloc(n  * sizeof(Point));
    struct Node *points_out = (struct Node *) malloc(n  * sizeof(Node));
    struct Node *expected_points = (struct Node *) malloc(n * sizeof(Node));

    srand((int)time(NULL));

    for (i = 0; i < n; ++i)
    {
        struct Node t2;
        struct Point t;
        temp = (float) n - i - 1;

        t.p[0] = temp, t.p[1] = temp, t.p[2] = temp;
        t2.p[0] = (float)i, t2.p[1] = (float)i, t2.p[2] = (float)i;

        points[i] = t;
        expected_points[i] = t2;
    }

    buildKdTree(points, n , points_out);

    // cuPrintTree(points_out, 0, n / 2);
    ASSERT_KD_TREE(points_out, n);

    free(points);
    free(points_out);
    free(expected_points);
}

TEST(kd_tree_build, correctness)
{
    int n;
    for (n = 2000; n <= 200000; n += 10000)
    {

        struct Point *points = (struct Point *) malloc(n  * sizeof(Point));
        struct Node *points_out = (struct Node *) malloc(n  * sizeof(Node));

        if (n > 10000)
        {
            populatePointSRosetta(points,  n);
            // readPoints("/home/simenhg/workspace/tsi-gpgpu/tests/data/100_mill_points.data", n, points);
        }
        else
        {
            readPoints("../tests/data/10000_points.data", n, points);

        }

        buildKdTree(points, n , points_out);
        // cuPrintTree(points_out, 0, n / 2);

        ASSERT_KD_TREE(points_out, n);

        free(points);
        free(points_out);

    }
}

TEST(kd_tree_build, timing)
{
    int n;

    // for (n = 8388608; n <= 8388608 ; n += 250000)
    for (n = 1024; n <= 1024 ; n += 250000)
    {
        struct Point *points = (struct Point *) malloc(n * sizeof(Point));
        struct Node *points_out = (struct Node *) malloc(n * sizeof(Node));

        populatePointSs(points, n);
        populatePoints(points_out, n);

        float elapsed_time;
        int bytes = n * (sizeof(Point));
        cudaEvent_t start, stop;

        cudaStartTiming(start, stop, elapsed_time);
        buildKdTree(points, n, points_out);
        cudaStopTiming(start, stop, elapsed_time);
        printCudaTiming(elapsed_time, bytes, n);

        free(points);
        free(points_out);
        cudaDeviceReset();
    }
}

TEST(kd_tree_build, wikipedia_example)
{
    cudaDeviceReset();
    int n = 6;
    struct Point *points = (struct Point *) malloc(n  * sizeof(Point));
    struct Node *points_out = (struct Node *) malloc(n  * sizeof(Node));
    struct Point *points_correct = (struct Point *) malloc(n  * sizeof(Point));

    points[0].p[0] = 2, points[0].p[1] = 3, points[0].p[2] = 0;
    points[1].p[0] = 5, points[1].p[1] = 4, points[1].p[2] = 0;
    points[2].p[0] = 9, points[2].p[1] = 6, points[2].p[2] = 0;
    points[3].p[0] = 4, points[3].p[1] = 7, points[3].p[2] = 0;
    points[4].p[0] = 8, points[4].p[1] = 1, points[4].p[2] = 0;
    points[5].p[0] = 7, points[5].p[1] = 2, points[5].p[2] = 0;

    buildKdTree(points, n, points_out);
    // cuPrintTree(points_out, 0, n / 2);

    points_correct[0].p[0] = 2, points_correct[0].p[1] = 3, points_correct[0].p[2] = 0;
    points_correct[1].p[0] = 5, points_correct[1].p[1] = 4, points_correct[1].p[2] = 0;
    points_correct[2].p[0] = 4, points_correct[2].p[1] = 7, points_correct[2].p[2] = 0;
    points_correct[3].p[0] = 7, points_correct[3].p[1] = 2, points_correct[3].p[2] = 0;
    points_correct[4].p[0] = 8, points_correct[4].p[1] = 1, points_correct[4].p[2] = 0;
    points_correct[5].p[0] = 9, points_correct[5].p[1] = 6, points_correct[5].p[2] = 0;

    ASSERT_KD_TREE(points_out, n);

    for (int i = 0; i < n; ++i)
    {
        ASSERT_EQ(points_correct[i].p[0], points_out[i].p[0]) << "failed at i = " << i;
        ASSERT_EQ(points_correct[i].p[1], points_out[i].p[1]) << "failed at i = " << i;
        ASSERT_EQ(points_correct[i].p[2], points_out[i].p[2]) << "failed at i = " << i;
    }
    free(points_out);
    free(points);
    free(points_correct);
}

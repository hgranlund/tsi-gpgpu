#include <knn_gpgpu.h>
#include <point.h>

#include <stdio.h>
#include <helper_cuda.h>
#include "gtest/gtest.h"

void populatePoints(Point *points, int n)
{
    int i;
    float temp;
    srand(time(NULL));

    for (i = 0; i < n; ++i)
    {
        Point t;
        temp = n - i - 1;

        t.p[0] = temp, t.p[1] = temp, t.p[2] = temp;

        points[i] = t;
    }
}
void populatePointSs(PointS *points, int n)
{
    int i;
    float temp;
    srand(time(NULL));

    for (i = 0; i < n; ++i)
    {
        PointS t;
        temp = n - i - 1;

        t.p[0] = temp, t.p[1] = temp, t.p[2] = temp;

        points[i] = t;
    }
}

void cudaStartTiming(cudaEvent_t &start, cudaEvent_t &stop, float &elapsed_time)
{
    elapsed_time = 0;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventRecord(start, 0));
}

void cudaStopTiming(cudaEvent_t &start, cudaEvent_t &stop, float &elapsed_time)
{
    checkCudaErrors(cudaEventRecord(stop, 0));
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
}

void printCudaTiming(float elapsed_time, float bytes, int n)
{
    double throughput = 1.0e-9 * ((double)bytes) / (elapsed_time * 1e-3);

    printf("Throughput = %.4f GB/s, Time = %.5f ms, Size = %u Elements\n", throughput, elapsed_time, n);
}

void ASSERT_TREE_EQ(Point *expected_tree, Point *actual_tree, int n)
{
    int i, j;

    for (i = 0; i < n; ++i)
    {
        for (j = 0; j < 3; ++j)
        {
            ASSERT_EQ(expected_tree[i].p[j] , actual_tree[i].p[j]) << "failed with i = " << i << " j = " << j ;
        }
    }
}

TEST(kd_tree_build, correctness)
{
    int i, n = 8;
    float temp;

    struct PointS *points = (PointS *) malloc(n  * sizeof(PointS));
    struct Point *points_out = (Point *) malloc(n  * sizeof(Point));
    struct Point *expected_points = (Point *) malloc(n * sizeof(Point));

    srand(time(NULL));

    for (i = 0; i < n; ++i)
    {
        Point t2;
        PointS t;
        temp = n - i - 1;

        t.p[0] = temp, t.p[1] = temp, t.p[2] = temp;
        t2.p[0] = i, t2.p[1] = i, t2.p[2] = i;

        points[i] = t;
        expected_points[i] = t2;
    }

    build_kd_tree(points, n , points_out);

    ASSERT_TREE_EQ(points_out, expected_points, n);

    free(points);
    free(points_out);
    free(expected_points);
}

TEST(kd_tree_build, timing)
{
    int n;

    // for (n = 8388608; n <= 8388608 ; n += 250000)
    for (n = 1024; n <= 1024 ; n += 250000)
    {
        struct PointS *points = (PointS *) malloc(n * sizeof(PointS));
        struct Point *points_out = (Point *) malloc(n * sizeof(Point));

        populatePointSs(points, n);
        populatePoints(points_out, n);

        float elapsed_time;
        int bytes = n * (sizeof(PointS));
        cudaEvent_t start, stop;

        cudaStartTiming(start, stop, elapsed_time);
        build_kd_tree(points, n, points_out);
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
    struct PointS *points = (PointS *) malloc(n  * sizeof(PointS));
    struct Point *points_out = (Point *) malloc(n  * sizeof(Point));
    struct PointS *points_correct = (PointS *) malloc(n  * sizeof(PointS));

    points[0].p[0] = 2, points[0].p[1] = 3, points[0].p[2] = 0;
    points[1].p[0] = 5, points[1].p[1] = 4, points[1].p[2] = 0;
    points[2].p[0] = 9, points[2].p[1] = 6, points[2].p[2] = 0;
    points[3].p[0] = 4, points[3].p[1] = 7, points[3].p[2] = 0;
    points[4].p[0] = 8, points[4].p[1] = 1, points[4].p[2] = 0;
    points[5].p[0] = 7, points[5].p[1] = 2, points[5].p[2] = 0;

    build_kd_tree(points, n, points_out);

    points_correct[0].p[0] = 2, points_correct[0].p[1] = 3, points_correct[0].p[2] = 0;
    points_correct[1].p[0] = 5, points_correct[1].p[1] = 4, points_correct[1].p[2] = 0;
    points_correct[2].p[0] = 4, points_correct[2].p[1] = 7, points_correct[2].p[2] = 0;
    points_correct[3].p[0] = 7, points_correct[3].p[1] = 2, points_correct[3].p[2] = 0;
    points_correct[4].p[0] = 8, points_correct[4].p[1] = 1, points_correct[4].p[2] = 0;
    points_correct[5].p[0] = 9, points_correct[5].p[1] = 6, points_correct[5].p[2] = 0;

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
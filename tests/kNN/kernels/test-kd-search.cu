#include "test-common.cuh"
#include <knn_gpgpu.h>

TEST(kd_search, wikipedia_example)
{
    int n = 6, k = 1;
    struct PointS *points = (struct PointS *) malloc(n * sizeof(PointS));
    struct Point *points_out = (struct Point *) malloc(n * sizeof(Point));
    int *result = (int *) malloc(n * k * sizeof(int));

    points[0].p[0] = 2, points[0].p[1] = 3, points[0].p[2] = 0;
    points[1].p[0] = 5, points[1].p[1] = 4, points[1].p[2] = 0;
    points[2].p[0] = 9, points[2].p[1] = 6, points[2].p[2] = 0;
    points[3].p[0] = 4, points[3].p[1] = 7, points[3].p[2] = 0;
    points[4].p[0] = 8, points[4].p[1] = 1, points[4].p[2] = 0;
    points[5].p[0] = 7, points[5].p[1] = 2, points[5].p[2] = 0;

    cudaDeviceReset();
    build_kd_tree(points, n, points_out);
    queryAll(points_out, points_out, n, n, 1, result);

    for (int i = 0; i < n; ++i)
    {
        ASSERT_EQ(result[i], i);
    }

    free(points);
    free(points_out);
    free(result);
}

TEST(kd_search, timing)
{
    int n, k = 1;

    for (n = 32; n <= 32; n += 250000)
    {
        struct PointS *points = (struct PointS *) malloc(n  * sizeof(PointS));
        struct Point *points_out = (struct Point *) malloc(n  * sizeof(Point));
        srand(time(NULL));

        populatePointSs(points, n);

        build_kd_tree(points, n, points_out);

        int test_runs = n;
        int *result = (int *) malloc(test_runs * k * sizeof(int));
        struct Point *query_data = (struct Point *) malloc(test_runs * sizeof(Point));

        populatePoints(query_data, n);

        cudaEvent_t start, stop;
        float elapsed_time = 0;
        int bytes = n * (sizeof(Point));

        cudaStartTiming(start, stop, elapsed_time);
        queryAll(query_data, points_out, test_runs, n, k, result);
        cudaStopTiming(start, stop, elapsed_time);
        printCudaTiming(elapsed_time, bytes, n);

        free(query_data);
        free(points_out);
        free(points);
        cudaDeviceReset();
    };
};

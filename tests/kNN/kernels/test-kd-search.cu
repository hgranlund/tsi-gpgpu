#include <kd-tree-naive.cuh>
#include <kd-search.cuh>
#include <knn_gpgpu.h>
#include <point.h>

#include <stdio.h>
#include <helper_cuda.h>
#include "gtest/gtest.h"

// TEST(kd_search, kd_search_correctness){
//     int i, j, n = 8;
//     float temp;
//     Point *points, *expected_points;
//     points = (Point*) malloc(n  * sizeof(Point));
//     expected_points = (Point*) malloc(n * sizeof(Point));
//     srand(time(NULL));

//     build_kd_tree(points, n);

//     for ( i = 0; i < n; ++i)
//     {
//         for ( j = 0; j < 3; ++j)
//         {
//             ASSERT_EQ(points[i].p[j] ,expected_points[i].p[j]) << "Faild with i = " << i << " j = " <<j ;
//         }
//     }
//     free(points);
//     free(expected_points);
// }

TEST(kd_search, kd_search_basic){
    int i, n = 5100000;
    Point *points;
    points = (Point*) malloc(n  * sizeof(Point));
    srand(time(NULL));

    for (i = 0; i < n; ++i)
    {
        Point t;
        t.p[0] = rand();
        t.p[1] = rand();
        t.p[2] = rand();
        points[i] = t;
    }

    cudaDeviceReset();
    cudaEvent_t start, stop;
    unsigned int bytes = n * (sizeof(Point));
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    float elapsed_time=0;

    checkCudaErrors(cudaEventRecord(start, 0));

    build_kd_tree(points, n);

    checkCudaErrors(cudaEventRecord(stop, 0));
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    elapsed_time = elapsed_time;
    double throughput = 1.0e-9 * ((double)bytes)/(elapsed_time* 1e-3);

    printf("Built kd-tree, throughput = %.4f GB/s, time = %.5f ms, n = %u elements\n",throughput, elapsed_time, n);

    store_locations(points, 0, n, n);

    int test_runs = 10000;
    float **query_data = (float**) malloc(test_runs * sizeof *query_data);

    for (i = 0; i < test_runs; i++)
    {
        query_data[i] = (float*) malloc(3 * sizeof *query_data[i]);
        query_data[i][0] = rand() % 1000;
        query_data[i][1] = rand() % 1000;
        query_data[i][2] = rand() % 1000;
    }

    for (i = 0; i < test_runs; i++) {
        // nn(query_data[i], points, dists, 0, midpoint(0, n));
        nn(query_data[i], points, 0, mid(0, n));
    }

    for (i = 0; i < test_runs; ++i)
    {
        free(query_data[i]);
    }
    free(query_data);
    
    free(points);
}
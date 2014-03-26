#include <kd-tree-naive.cuh>
#include <kd-search.cuh>
#include <knn_gpgpu.h>
#include <point.h>

#include <stdio.h>
#include <helper_cuda.h>
#include "gtest/gtest.h"

// void ASSERT_QUERY_EQ(struct Point *tree, int n, float qx, float qy, float qz, float ex, float ey, float ez)
// {
//     float dists[n];

//     float query_point[3];
//     query_point[0] = qx, query_point[1] = qy, query_point[2] = qz;

//     // int best_fit = nn(query_point, tree, dists, 0, midpoint(0, n));
//     int mid = (int) floor((n) / 2);
//     int best_fit = nn(query_point, tree, 0, mid);

//     float actual = tree[best_fit].p[0] + tree[best_fit].p[1] + tree[best_fit].p[2];
//     float expected = ex + ey + ez;

//     ASSERT_EQ(actual, expected);
// }

// TEST(kd_search, kd_search_wiki_correctness){
//     int i, wn = 6;
//     struct Point *wiki = (Point*) malloc(wn  * sizeof(Point));

//     // (2,3), (5,4), (9,6), (4,7), (8,1), (7,2).
//     wiki[0].p[0] = 2, wiki[0].p[1] = 3, wiki[0].p[2] = 0;
//     wiki[1].p[0] = 5, wiki[1].p[1] = 4, wiki[1].p[2] = 0;
//     wiki[2].p[0] = 9, wiki[2].p[1] = 6, wiki[2].p[2] = 0;
//     wiki[3].p[0] = 4, wiki[3].p[1] = 7, wiki[3].p[2] = 0;
//     wiki[4].p[0] = 8, wiki[4].p[1] = 1, wiki[4].p[2] = 0;
//     wiki[5].p[0] = 7, wiki[5].p[1] = 2, wiki[5].p[2] = 0;

//     cudaDeviceReset();
//     build_kd_tree(wiki, wn);
//     store_locations(wiki, 0, wn, wn);

//     ASSERT_QUERY_EQ(wiki, wn, 2, 3, 0, 2, 3, 0);
//     ASSERT_QUERY_EQ(wiki, wn, 5, 4, 0, 5, 4, 0);
//     ASSERT_QUERY_EQ(wiki, wn, 9, 6, 0, 9, 6, 0);
//     ASSERT_QUERY_EQ(wiki, wn, 4, 7, 0, 4, 7, 0);
//     ASSERT_QUERY_EQ(wiki, wn, 8, 1, 0, 8, 1, 0);
//     ASSERT_QUERY_EQ(wiki, wn, 7, 2, 0, 7, 2, 0);
//     ASSERT_QUERY_EQ(wiki, wn, 10, 10, 0, 9, 6, 0);
//     ASSERT_QUERY_EQ(wiki, wn, 0, 0, 0, 2, 3, 0);
//     ASSERT_QUERY_EQ(wiki, wn, 4, 4, 0, 5, 4, 0);
//     ASSERT_QUERY_EQ(wiki, wn, 3, 2, 0, 2, 3, 0);
//     ASSERT_QUERY_EQ(wiki, wn, 2, 6, 0, 4, 7, 0);
//     ASSERT_QUERY_EQ(wiki, wn, 10, 0, 0, 8, 1, 0);
//     ASSERT_QUERY_EQ(wiki, wn, 0, 10, 0, 4, 7, 0);
// }

// TEST(kd_search, kd_search_timing){
//     int i, n = 5100000;
//     Point *points;
//     points = (Point*) malloc(n  * sizeof(Point));
//     srand(time(NULL));

//     for (i = 0; i < n; ++i)
//     {
//         Point t;
//         t.p[0] = rand();
//         t.p[1] = rand();
//         t.p[2] = rand();
//         points[i] = t;
//     }

//     cudaDeviceReset();
//     cudaEvent_t start, stop;
//     unsigned int bytes = n * (sizeof(Point));
//     checkCudaErrors(cudaEventCreate(&start));
//     checkCudaErrors(cudaEventCreate(&stop));
//     float elapsed_time=0;

//     checkCudaErrors(cudaEventRecord(start, 0));

//     build_kd_tree(points, n);

//     checkCudaErrors(cudaEventRecord(stop, 0));
//     cudaEventSynchronize(start);
//     cudaEventSynchronize(stop);
//     cudaEventElapsedTime(&elapsed_time, start, stop);
//     elapsed_time = elapsed_time;
//     double throughput = 1.0e-9 * ((double)bytes)/(elapsed_time* 1e-3);

//     // printf("Built kd-tree, throughput = %.4f GB/s, time = %.5f ms, n = %u elements\n",throughput, elapsed_time, n);

//     store_locations(points, 0, n, n);

//     int test_runs = 10000;
//     float **query_data = (float**) malloc(test_runs * sizeof *query_data);

//     for (i = 0; i < test_runs; i++)
//     {
//         query_data[i] = (float*) malloc(3 * sizeof *query_data[i]);
//         query_data[i][0] = rand() % 1000;
//         query_data[i][1] = rand() % 1000;
//         query_data[i][2] = rand() % 1000;
//     }

//     for (i = 0; i < test_runs; i++) {
//         // nn(query_data[i], points, dists, 0, midpoint(0, n));
//         int mid = (int) floor((n) / 2);
//         nn(query_data[i], points, 0, mid);
//     }

//     for (i = 0; i < test_runs; ++i)
//     {
//         free(query_data[i]);
//     }
//     free(query_data);

//     free(points);
// }

TEST(kd_search, kd_search_all_points)
{
    int i,
        n = 100000,
        n_qp = n,
        k = 1,
        *result;
    Point *points;
    points = (Point *) malloc(n  * sizeof(Point));
    result = (int *) malloc(n_qp  * k * sizeof(int));
    srand(time(NULL));

    // for (i = 0; i < n; ++i)
    // {
    //     Point t;
    //     t.p[0] = rand();
    //     t.p[1] = rand();
    //     t.p[2] = rand();
    //     points[i] = t;
    // }

    // cudaDeviceReset();
    // build_kd_tree(points, n);
    // store_locations(points, 0, n, n);

    cudaDeviceReset();
    cudaEvent_t start, stop;
    unsigned int bytes = n * (sizeof(Point));
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    float elapsed_time = 0;

    // checkCudaErrors(cudaEventRecord(start, 0));

    queryAll(points, points, n, n, k, result);

    checkCudaErrors(cudaEventRecord(stop, 0));
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    elapsed_time = elapsed_time;
    double throughput = 1.0e-9 * ((double)bytes) / (elapsed_time * 1e-3);

    printf("Searched for n queries, throughput = %.4f GB/s, time = %.5f ms, n = %u elements\n", throughput, elapsed_time, n);

    free(points);
}

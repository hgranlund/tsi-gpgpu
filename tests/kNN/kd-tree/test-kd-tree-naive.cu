#include <kd-tree-naive.cuh>
#include <knn_gpgpu.h>
#include <point.h>

#include <stdio.h>
#include <helper_cuda.h>
#include "gtest/gtest.h"

#define debug 0

__host__  void h_printPointsArray__(Point *l, int n, char *s, int l_debug = 0)
{
    if (debug || l_debug)
    {
        printf("%10s: [ ", s);
        for (int i = 0; i < n; ++i)
        {
            printf("%3.1f, ", l[i].p[0]);
        }
        printf("]\n");
    }
}


int h_index(int i, int j, int n)
{
    return i + j * n;
}

void h_swap(Point *points, int a, int b, int n)
{
    Point t = points[a];
    points[a] = points[b], points[b] = t;
}

int midpoint(int lower, int upper)
{
    return (int) floor((float)(upper - lower) / 2) + lower;
}

void print_tree(Point *tree, int level, int lower, int upper, int n)
{
    if (debug)
    {
        if (lower >= upper)
        {
            return;
        }

        int i, r = midpoint(lower, upper);

        printf("|");
        for (i = 0; i < level; ++i)
        {
            printf("--");
        }
        printf("(%3.1f, %3.1f, %3.1f)\n", tree[r].p[0], tree[r].p[1], tree[r].p[2]);

        print_tree(tree, 1 + level, lower, r, n);
        print_tree(tree, 1 + level, r + 1, upper, n);
    }
}

void _print_t(Point *tree, int level, int lower, int upper, int n)
{
    if (lower >= upper)
    {
        return;
    }

    int i, r = floor((upper - lower) / 2) + lower;

    printf("|");
    for (i = 0; i < level; ++i)
    {
        printf("--");
    }
    printf("(%3.1f, %3.1f, %3.1f)\n", tree[r].p[0], tree[r].p[1], tree[r].p[2]);

    _print_t(tree, 1 + level, lower, r, n);
    _print_t(tree, 1 + level, r + 1, upper, n);
}

TEST(kd_tree_naive, kd_tree_naive_correctness)
{
    int i, j, n = 8;
    float temp;
    Point *points, *expected_points;
    points = (Point *) malloc(n  * sizeof(Point));
    expected_points = (Point *) malloc(n * sizeof(Point));
    srand(time(NULL));
    for ( i = 0; i < n; ++i)
    {
        temp = n - i - 1;
        Point t;
        t.p[0] = temp;
        t.p[1] = temp;
        t.p[2] = temp;
        points[i]    = t;
        Point t2;
        t2.p[0] = i;
        t2.p[1] = i;
        t2.p[2] = i;
        expected_points[i] = t2;
    }
    if (debug)
    {
        printf("kd tree expected:\n");
        print_tree(expected_points, 0, 0, n, n);
        printf("==================\n");

        printf("kd tree:\n");
        print_tree(points, 0, 0, n, n);
        printf("==================\n");

        for (int i = 0; i < n; ++i)
        {
            printf("%3.1f, ", points[i].p[0]);
        }
        printf("\n");
    }

    build_kd_tree(points, n);

    // h_printPointsArray__(points, n, "points coplete", 0);

    for ( i = 0; i < n; ++i)
    {
        for ( j = 0; j < 3; ++j)
        {
            ASSERT_EQ(points[i].p[j] , expected_points[i].p[j]) << "Faild with i = " << i << " j = " << j ;
        }
    }
    free(points);
    free(expected_points);
}


TEST(kd_tree_naive, kd_tree_naive_time)
{
    int i, n = 8388608;
    // for (n = 13000000; n <=13000000 ; n+=250000)
    for (n = 8388608; n <= 8388608 ; n += 250000)
    {
        cudaDeviceReset();
        float temp;
        Point *points;
        points = (Point *) malloc(n  * sizeof(Point));
        srand(time(NULL));
        for ( i = 0; i < n; ++i)
        {
            temp = n - i - 1;
            Point t;
            t.p[0] = temp;
            t.p[1] = temp;
            t.p[2] = temp;
            points[i]    = t;

        }

        cudaEvent_t start, stop;
        unsigned int bytes = n * (sizeof(Point));
        checkCudaErrors(cudaEventCreate(&start));
        checkCudaErrors(cudaEventCreate(&stop));
        float elapsed_time = 0;

        checkCudaErrors(cudaEventRecord(start, 0));

        build_kd_tree(points, n);

        checkCudaErrors(cudaEventRecord(stop, 0));
        cudaEventSynchronize(start);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_time, start, stop);
        elapsed_time = elapsed_time ;
        double throughput = 1.0e-9 * ((double)bytes) / (elapsed_time * 1e-3);
        printf("build_kd_tree_naive, Throughput = %.4f GB/s, Time = %.5f ms, Size = %u Elements, NumDevsUsed = %d\n",
               throughput, elapsed_time, n, 1);
        free(points);
    }
}


///////////////////////////////////////////////
// Failing spec from Teodor

TEST(kd_tree_naive, wikipedia_exsample)
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

    cudaDeviceReset();

    h_printPointsArray__(wiki, wn, "Wikipedia", 1);
    build_kd_tree(wiki, wn);
    _print_t(wiki, 0, 0, wn, wn);
    printf("\n");

    h_printPointsArray__(wiki, wn, "Wikipedia", 1);
    printf("\n");

    // _build_kd_tree(wiki, wn);
    // _print_t(wiki, 0, 0, wn, wn);
    // printf("\n");
}


// TEST(kd_tree_naive, kd_tree_naive_step_time){
//   int i, n, p, numBlocks, numThreads, *d_partition;
//   float temp;
//   Point *h_points;
//   srand(time(NULL));
//   p = 65536;
//   n= 4 *p;
//   h_points = (Point*) malloc(n  * sizeof(Point));
//   for ( i = 0; i < n; ++i)
//   {
//     temp = n-i-1;
//     h_points[i] =(Point) {.p={temp,temp,temp}};
//   }
//   Point *d_points, *d_swap;

//   checkCudaErrors(
//     cudaMalloc(&d_partition, n*sizeof(int)));

//   checkCudaErrors(
//     cudaMalloc(&d_points, n*sizeof(Point)));

//   checkCudaErrors(
//     cudaMalloc(&d_swap, n*sizeof(Point)));

//   checkCudaErrors(
//     cudaMemcpy(d_points, h_points, n*sizeof(Point), cudaMemcpyHostToDevice));

//   cudaEvent_t start, stop;
//   unsigned int bytes = n * (sizeof(Point));
//   checkCudaErrors(cudaEventCreate(&start));
//   checkCudaErrors(cudaEventCreate(&stop));
//   float elapsed_time=0;

//   checkCudaErrors(cudaEventRecord(start, 0));


//   getThreadAndBlockCountMulRadix(n, p, numBlocks, numThreads);
//   cuBalanceBranch<<<numBlocks,numThreads>>>(d_points, d_swap, d_partition, n/p, p, 0);

//   checkCudaErrors(cudaEventRecord(stop, 0));
//   cudaEventSynchronize(start);
//   cudaEventSynchronize(stop);
//   cudaEventElapsedTime(&elapsed_time, start, stop);
//   elapsed_time = elapsed_time ;
//   double throughput = 1.0e-9 * ((double)bytes)/(elapsed_time* 1e-3);
//   printf("kd_tree_naive_step, Throughput = %.4f GB/s, Time = %.5f ms, Size = %u, p = %d Elements, NumDevsUsed = %d\n",
//     throughput, elapsed_time, n/p, p, 1);

//   checkCudaErrors(
//     cudaMemcpy(h_points, d_points, n*sizeof(Point), cudaMemcpyDeviceToHost));


//   free(h_points);
//   checkCudaErrors(cudaFree(d_points));
//   checkCudaErrors(cudaFree(d_swap));
//   checkCudaErrors(cudaFree(d_partition));

// }

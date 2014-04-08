#include <kd-tree-build.cuh>
#include <knn_gpgpu.h>
#include <point.h>

#include <stdio.h>
#include <helper_cuda.h>
#include "gtest/gtest.h"

#define debug 0

__host__  void h_printPointsArray__(PointS *l, int n, char *s, int l_debug = 0)
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

void h_swap(PointS *points, int a, int b, int n)
{
    PointS t = points[a];
    points[a] = points[b], points[b] = t;
}

int midpoint(int lower, int upper)
{
    return (int) floor((float)(upper - lower) / 2) + lower;
}

void print_tree(PointS *tree, int level, int lower, int upper, int n)
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

void _print_t(PointS *tree, int level, int lower, int upper, int n)
{
    if (debug)
    {
        if (lower >= upper)
        {
            return;
        }

        int i, r = floor((float)(upper - lower) / 2) + lower;

        printf("|");
        for (i = 0; i < level; ++i)
        {
            printf("--");
        }
        printf("(%3.1f, %3.1f, %3.1f)\n", tree[r].p[0], tree[r].p[1], tree[r].p[2]);

        _print_t(tree, 1 + level, lower, r, n);
        _print_t(tree, 1 + level, r + 1, upper, n);
    }
}

TEST(kd_tree_naive, kd_tree_naive_correctness)
{
    int i, j, n = 8;
    float temp;
    PointS *points, *expected_points;
    Point *points_out;
    points = (PointS *) malloc(n  * sizeof(PointS));
    points_out = (Point *) malloc(n  * sizeof(Point));
    expected_points = (PointS *) malloc(n * sizeof(PointS));
    srand(time(NULL));
    for ( i = 0; i < n; ++i)
    {
        temp = n - i - 1;
        PointS t;
        t.p[0] = temp;
        t.p[1] = temp;
        t.p[2] = temp;
        points[i]    = t;
        PointS t2;
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

    build_kd_tree(points, n , points_out);

    // h_printPointsArray__(points_out, n, "points coplete", 0);

    for ( i = 0; i < n; ++i)
    {
        for ( j = 0; j < 3; ++j)
        {
            ASSERT_EQ(points_out[i].p[j] , expected_points[i].p[j]) << "Faild with i = " << i << " j = " << j ;
        }
    }
    free(points);
    free(points_out);
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
        PointS *points;
        Point *points_out;
        points = (PointS *) malloc(n  * sizeof(PointS));
        points_out = (Point *) malloc(n  * sizeof(Point));
        srand(time(NULL));
        for ( i = 0; i < n; ++i)
        {
            temp = n - i - 1;
            PointS t;
            t.p[0] = temp;
            t.p[1] = temp;
            t.p[2] = temp;
            points[i]    = t;

        }

        cudaEvent_t start, stop;
        unsigned int bytes = n * (sizeof(PointS));
        checkCudaErrors(cudaEventCreate(&start));
        checkCudaErrors(cudaEventCreate(&stop));
        float elapsed_time = 0;

        checkCudaErrors(cudaEventRecord(start, 0));

        build_kd_tree(points, n, points_out);

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


TEST(kd_tree_naive, wikipedia_exsample)
{
    cudaDeviceReset();
    int wn = 6;
    struct PointS *wiki = (PointS *) malloc(wn  * sizeof(PointS));
    struct Point *wiki_out = (Point *) malloc(wn  * sizeof(Point));
    struct PointS *wiki_correct = (PointS *) malloc(wn  * sizeof(PointS));


    // (2,3), (5,4), (9,6), (4,7), (8,1), (7,2).
    wiki[0].p[0] = 2, wiki[0].p[1] = 3, wiki[0].p[2] = 0;
    wiki[1].p[0] = 5, wiki[1].p[1] = 4, wiki[1].p[2] = 0;
    wiki[2].p[0] = 9, wiki[2].p[1] = 6, wiki[2].p[2] = 0;
    wiki[3].p[0] = 4, wiki[3].p[1] = 7, wiki[3].p[2] = 0;
    wiki[4].p[0] = 8, wiki[4].p[1] = 1, wiki[4].p[2] = 0;
    wiki[5].p[0] = 7, wiki[5].p[1] = 2, wiki[5].p[2] = 0;


    build_kd_tree(wiki, wn, wiki_out);
    // _print_t(wiki, 0, 0, wn, wn);


    wiki_correct[0].p[0] = 2, wiki_correct[0].p[1] = 3, wiki_correct[0].p[2] = 0;
    wiki_correct[1].p[0] = 5, wiki_correct[1].p[1] = 4, wiki_correct[1].p[2] = 0;
    wiki_correct[2].p[0] = 4, wiki_correct[2].p[1] = 7, wiki_correct[2].p[2] = 0;
    wiki_correct[3].p[0] = 7, wiki_correct[3].p[1] = 2, wiki_correct[3].p[2] = 0;
    wiki_correct[4].p[0] = 8, wiki_correct[4].p[1] = 1, wiki_correct[4].p[2] = 0;
    wiki_correct[5].p[0] = 9, wiki_correct[5].p[1] = 6, wiki_correct[5].p[2] = 0;
    _print_t(wiki_correct, 0, 0, wn, wn);

    for (int i = 0; i < wn; ++i)

    {
        ASSERT_EQ(wiki_correct[i].p[0], wiki_out[i].p[0]) << "failed at i = " << i;
        ASSERT_EQ(wiki_correct[i].p[1], wiki_out[i].p[1]) << "failed at i = " << i;
        ASSERT_EQ(wiki_correct[i].p[2], wiki_out[i].p[2]) << "failed at i = " << i;
    }
    free(wiki_out);
    free(wiki);
    free(wiki_correct);
}

#include <float.h>
#include <math.h>

#include <test-common.cuh>

#include <kd-search.cuh>

#include <knn_gpgpu.h>


TEST(kd_search, query_all_correctness_with_10000_points_file)
{
    int n, i, k = 10;

    for (n = 100; n <= 100; n += 100000)
    {
        struct Point *points = (struct Point *) malloc(n  * sizeof(Point));
        struct Node *tree = (struct Node *) malloc(n  * sizeof(Node));

        srand((int)time(NULL));

        if (n > 10000)
        {
            populatePointSRosetta(points,  n);
            // readPoints("/home/simenhg/workspace/tsi-gpgpu/tests/data/100_mill_points.data", n, points);
        }
        else
        {
            readPoints("../tests/data/10000_points.data", n, points);
        }

        cudaDeviceReset();
        buildKdTree(points, n, tree);
        // printTree(tree, 0, n / 2);
        int *result = (int *) malloc(n * k * sizeof(int));

        queryAll(points, tree, n, n, k, result, 100);

        for (i = 0; i < n; ++i)
        {
            quickSortResult(result + (i * k), tree, points[i], k);
            ASSERT_GT(result[i * k], -1) << "Result index is less then 0 \n Failed at i = " << i << " with n = " << n ;
            ASSERT_LT(result[i * k], n) << "Result index is bigger then the length of the tree \n Failed at i = " << i << " with n = " << n ;
            ASSERT_EQ(points[i].p[0], tree[result[i * k]].p[0]) << "Failed at i = " << i << " with n = " << n ;
            ASSERT_EQ(points[i].p[1], tree[result[i * k]].p[1]) << "Failed at i = " << i << " with n = " << n;
            ASSERT_EQ(points[i].p[2], tree[result[i * k]].p[2]) << "Failed at i = " << i << " with n = " << n;
        }

        free(tree);
        free(result);
        free(points);
    };
};

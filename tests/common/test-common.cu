#include "test-common.cuh"
#include "math.h"
#include <sys/time.h>

void populatePoints(struct Point *points, int n)
{
    int i;
    float temp;
    srand(time(NULL));

    for (i = 0; i < n; ++i)
    {
        struct Point t;
        temp = n - i - 1;

        t.p[0] = temp, t.p[1] = temp, t.p[2] = temp;

        points[i] = t;
    }
}

void populatePointSs(struct PointS *points, int n)
{
    int i;
    float temp;
    srand(time(NULL));

    for (i = 0; i < n; ++i)
    {
        struct PointS t;
        temp = n - i - 1;

        t.p[0] = temp, t.p[1] = temp, t.p[2] = temp;

        points[i] = t;
    }
}

#define rand1() (rand() / (double)RAND_MAX)

void populatePointSRosetta(struct PointS *points, int n)
{
    int i;
    srand(time(NULL));

    for (i = 0; i < n; ++i)
    {
        struct PointS t;
        t.p[0] = rand1(), t.p[1] = rand1(), t.p[2] = rand1();
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

double WallTime ()
{
    struct timeval tmpTime;
    gettimeofday(&tmpTime, NULL);
    return tmpTime.tv_sec + tmpTime.tv_usec / 1.0e6;
}

void printTree(struct Point *tree, int level, int root)
{
    if (root < 0) return;

    int i;

    printf("|");
    for (i = 0; i < level; ++i)
    {
        printf("----");
    }
    printf("(%3.1f, %3.1f, %3.1f)\n", tree[root].p[0], tree[root].p[1], tree[root].p[2]);

    printTree(tree, 1 + level, tree[root].left);
    printTree(tree, 1 + level, tree[root].right);
}

void readPoints(const char *file_path, int n, struct PointS *points)
{
    FILE *file = fopen(file_path, "rb");
    if (file == NULL)
    {
        fputs ("File error\n", stderr);
        exit (1);
    }
    for (int i = 0; i < n; ++i)
    {
        fread(&points[i].p, sizeof(float), 3, file);
        for (int j = 0; j < 3; ++j)
        {
            points[i].p[j] = round(points[i].p[j] / 100000000.0);
        }
    }

    fclose(file);
}

void ASSERT_TREE_EQ(struct Point *expected_tree, struct Point *actual_tree, int n)
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


void ASSERT_TREE_LEVEL_OK(struct PointS *points, int *steps, int n, int p, int dim)
{
    struct PointS *t_points;

    for (int i = 0; i < p; ++i)
    {
        t_points = points + steps[i * 2];
        n =  steps[i * 2 + 1] - steps[i * 2];

        for (int i = 0; i < n / 2; ++i)
        {
            ASSERT_LE(t_points[i].p[dim], t_points[n / 2].p[dim]) << "Faild with n = " << n << " and p " << p << " at i = " << i << " with dim = " << dim;
        }

        for (int i = (n / 2) ; i < n; ++i)
        {
            ASSERT_GE(t_points[i].p[dim], t_points[n / 2].p[dim]) << "Faild with n = " << n << " and p " << p << " at i =" << i << " with dim = " << dim;
        }
    }
}

void ASSERT_KD_TREE_LEVEL(struct Point *tree, int dim, int lower, int upper, int n)
{

    int i,
        mid = lower + ((upper - lower) / 2);

    struct Point piv = tree[mid];

    if (piv.left == -1 || piv.right == -1)
    {
        if (piv.right != -1)
        {
            ASSERT_GE(tree[piv.right].p[dim], piv.p[dim]) << "Failed with n = " << n << " at leaf \nPivot (" << piv.p[0] << ", " << piv.p[1] << ", " << piv.p[2] << ") failed on greater than (" << tree[piv.right].p[0] << ", " << tree[piv.right].p[1] << ", " << tree[piv.right].p[2] << ")";
        }
        else if (piv.left != -1)
        {
            ASSERT_LE(tree[piv.left].p[dim], piv.p[dim]) << "Failed with n = " << n << " at leaf \nPivot (" << piv.p[0] << ", " << piv.p[1] << ", " << piv.p[2] << ") failed on less than (" << tree[piv.left].p[0] << ", " << tree[piv.left].p[1] << ", " << tree[piv.left].p[2] << ")";
        }
        return;
    }

    for (i = lower; i < mid; ++i)
    {
        ASSERT_LE(tree[i].p[dim], piv.p[dim]) << "Failed with n = " << n << "\nPivot (" << piv.p[0] << ", " << piv.p[1] << ", " << piv.p[2] << ") failed on less than (" << tree[i].p[0] << ", " << tree[i].p[1] << ", " << tree[i].p[2] << ")";
    }

    for (i = mid + 1; i < upper; ++i)
    {
        ASSERT_GE(tree[i].p[dim], piv.p[dim]) << "Failed with n = " << n << "\nPivot (" << piv.p[0] << ", " << piv.p[1] << ", " << piv.p[2] << ") failed on greater than (" << tree[i].p[0] << ", " << tree[i].p[1] << ", " << tree[i].p[2] << ")";
    }

    dim = (dim + 1) % 3;

    ASSERT_KD_TREE_LEVEL(tree, dim, lower, mid, n);
    ASSERT_KD_TREE_LEVEL(tree, dim, mid + 1, upper, n);
}

void ASSERT_KD_TREE(struct Point *tree, int n)
{
    ASSERT_KD_TREE_LEVEL(tree, 0, 0, n, n);
}

#include "kd-tree-naive.cuh"
#include "knn_gpgpu.h"
#include "point.h"

#include "stdio.h"
#include "helper_cuda.h"
#include "gtest/gtest.h"

#define debug 0

void writePoints(char *file_path, int n, PointS *points)
{
    printf("writing points...\n");

    FILE *file = fopen(file_path, "w");
    if (file == NULL)
    {
        fputs ("File error\n", stderr);
        exit (1);
    }
    for (int i = 0; i < n; ++i)
    {
        fwrite(&points[i].p, sizeof(float), 3, file);
    }
    fclose(file);
}


void readPoints(const char *file_path, int n, PointS *points)
{
    printf("Reading points...\n");

    FILE *file = fopen(file_path, "rb");
    if (file == NULL)
    {
        fputs ("File error\n", stderr);
        exit (1);
    }
    for (int i = 0; i < n; ++i)
    {
        fread(&points[i].p, sizeof(float), 3, file);
    }

    fclose(file);
}


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

void populatePoints(PointS *points, int n)
{
    srand(time(NULL));
    for (int i = 0; i < n; ++i)
    {
        PointS t;
        t.p[0] = rand();
        t.p[1] = rand();
        t.p[2] = rand();
        points[i]    = t;
    }
}

int main(int argc, char const *argv[])
{
    int n, nu, ni = 8388608,
               step = 250000;
    bool from_file = 0;
    n = nu = ni;
    if (argc == 2)
    {
        nu = ni = atoi(argv[1]);
        printf("Running kd-tree-build with n = %d\n", nu);
    }
    else if (argc == 3)
    {
        nu = ni = atoi(argv[1]);
        from_file = 1;
        printf("Running kd-tree-build from file '%s' with n = %d\n", argv[2], nu);
    }
    else if (argc == 4)
    {
        nu = atoi(argv[1]);
        ni = atoi(argv[2]);
        step = atoi(argv[3]);
        printf("Running kd-tree-build from n = %d to n = %d with step = %d\n", nu, ni, step);
    }
    else
    {
        printf("Running kd-tree-build with n = %d\n", nu);
    }

    for (n = nu; n <= ni ; n += step)
    {
        cudaDeviceReset();
        PointS *points;
        Point *points_out;
        points_out = (Point *) malloc(n  * sizeof(Point));
        points = (PointS *) malloc(n  * sizeof(PointS));

        if (from_file)
        {
            readPoints(argv[2], n, points);
        }
        else
        {
            populatePoints(points, n);
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
        free(points_out);
    }
    return 0;

}


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
    int n, nu, ni = 1024,
               step = 250000,
               k = 1;
    bool from_file = 0;
    n = nu = ni;
    if (argc == 2)
    {
        nu = ni = atoi(argv[1]);
        printf("Running kd-search-all with n = %d\n", nu);
    }
    else if (argc == 3)
    {
        nu = ni = atoi(argv[1]);
        from_file = 1;
        printf("Running kd-search-all from file '%s' with n = %d\n", argv[2], nu);
    }
    else if (argc == 4)
    {
        nu = atoi(argv[1]);
        ni = atoi(argv[2]);
        step = atoi(argv[3]);
        printf("Running kd-search-all from n = %d to n = %d with step = %d\n", nu, ni, step);
    }
    else
    {
        printf("Running kd-search-all with n = %d\n", nu);
    }

    for (n = nu; n <= ni ; n += step)
    {
        cudaDeviceReset();
        PointS *points;
        Point *points_out;
        points_out = (Point *) malloc(n  * sizeof(Point));
        points = (PointS *) malloc(n  * sizeof(PointS));
        int *result = (int *) malloc(n * k * sizeof(int));

        if (from_file)
        {
            readPoints(argv[2], n, points);
        }
        else
        {
            populatePoints(points, n);
        }
        cudaEvent_t start, stop;
        checkCudaErrors(cudaEventCreate(&start));
        checkCudaErrors(cudaEventCreate(&stop));
        float elapsed_time_build = 0;

        checkCudaErrors(cudaEventRecord(start, 0));

        build_kd_tree(points, n, points_out);

        checkCudaErrors(cudaEventRecord(stop, 0));
        cudaEventSynchronize(start);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_time_build, start, stop);


        cudaDeviceReset();
        checkCudaErrors(cudaEventCreate(&start));
        checkCudaErrors(cudaEventCreate(&stop));
        float elapsed_time_search = 0;

        checkCudaErrors(cudaEventRecord(start, 0));

        queryAll(points_out, points_out, n, n, k, result);

        checkCudaErrors(cudaEventRecord(stop, 0));
        cudaEventSynchronize(start);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_time_search, start, stop);

        printf("kd-search-all,  Build Time = %.5f ms, Query Time = %.5f ms, Total time = %.5f ms, Size = %u Elements, NumDevsUsed = %d\n",
               elapsed_time_build, elapsed_time_search, elapsed_time_build + elapsed_time_search, n, 1);

        free(points);
        free(result);
        free(points_out);
    }
    return 0;

}


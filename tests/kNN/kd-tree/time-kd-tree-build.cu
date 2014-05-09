#include "knn_gpgpu.h"
#include <stdio.h>
#include <helper_cuda.h>

void writePoints(char *file_path, int n, struct Point *points)
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

void readPoints(const char *file_path, int n, struct Point *points)
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

void populatePoints(struct Point *points, int n)
{
    int i;
    srand(time(NULL));

    for (i = 0; i < n; ++i)
    {
        struct Point t;
        t.p[0] = rand(), t.p[1] = rand(), t.p[2] = rand();
        points[i] = t;
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
        struct Node *points_out = (struct Node *) malloc(n  * sizeof(Node));
        struct Point *points = (struct Point *) malloc(n  * sizeof(Point));

        if (from_file)
        {
            readPoints(argv[2], n, points);
        }
        else
        {
            populatePoints(points, n);
        }

        cudaEvent_t start, stop;
        float elapsed_time = 0;
        checkCudaErrors(cudaEventCreate(&start));
        checkCudaErrors(cudaEventCreate(&stop));
        checkCudaErrors(cudaEventRecord(start, 0));

        buildKdTree(points, n, points_out);

        checkCudaErrors(cudaEventRecord(stop, 0));
        cudaEventSynchronize(start);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_time, start, stop);

        printf("buildKdTree_naive,  Time = %.5f ms, Size = %u Elements, NumDevsUsed = %d\n",
               elapsed_time, n, 1);

        free(points);
        free(points_out);
        cudaDeviceReset();
    }
    return 0;
}


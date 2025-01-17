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
    srand((int)time(NULL));
    for (int i = 0; i < n; ++i)
    {
        struct Point t;
        t.p[0] = (float) rand();
        t.p[1] = (float) rand();
        t.p[2] = (float) rand();
        points[i]    = t;
    }
}

int main(int argc, char const *argv[])
{
    int n, nu, ni = 1024,
               step = 250000,
               k = 1,
               no_of_runs = 1000;

    bool from_file = 0,
         variable_k = 0;

    n = nu = ni;

    if (argc == 2)
    {
        nu = ni = atoi(argv[1]);
        printf("Running kd-search-all with n = %d, k = %d\n", nu, k);
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
    else if (argc == 5)
    {
        nu = atoi(argv[1]);
        ni = atoi(argv[2]);
        step = atoi(argv[3]);
        k = atoi(argv[4]);
        variable_k = 1;
        printf("Running kd-search-%d from n = %d to n = %d with step = %d, k = %d\n", no_of_runs, nu, ni, step, k);
    }
    else
    {
        printf("Running kd-search-all with n = %d\n", nu);
    }

    for (n = nu; n <= ni ; n += step)
    {
        struct Node *tree = (struct Node *) malloc(n  * sizeof(Node));
        struct Point *points = (struct Point *) malloc(n  * sizeof(Point));
        int *result = (int *) malloc(n * k * sizeof(int));

        if (from_file)
        {
            readPoints(argv[2], n, points);
        }
        else
        {
            populatePoints(points, n);
			printf("*Alok - Finished Reading Points\n");
        }

        cudaEvent_t start, stop;
        float elapsed_time_build = 0;
        checkCudaErrors(cudaEventCreate(&start));
        checkCudaErrors(cudaEventCreate(&stop));
        checkCudaErrors(cudaEventRecord(start, 0));

        buildKdTree(points, n, tree);
		printf("*Alok - Finished Building Tree\n");

        checkCudaErrors(cudaEventRecord(stop, 0));
        cudaEventSynchronize(start);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_time_build, start, stop);

        cudaDeviceReset();

        float elapsed_time_search = 0;
        checkCudaErrors(cudaEventCreate(&start));
        checkCudaErrors(cudaEventCreate(&stop));
        checkCudaErrors(cudaEventRecord(start, 0));

        if (variable_k)
        {
            cuQueryAll(points, tree, no_of_runs, n, k, result);
        }
        else
        {
			printf("*Alok - Querying all...\n");
            cuQueryAll(points, tree, n, n, k, result);
			printf("*Alok - Querying all done\n");
        }

        checkCudaErrors(cudaEventRecord(stop, 0));
        cudaEventSynchronize(start);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_time_search, start, stop);

        printf("kd-search-all,  Build Time = %.5f ms, Query Time = %.5f ms, Total time = %.5f ms, Size = %u Elements, NumDevsUsed = %d\n",
               elapsed_time_build, elapsed_time_search, elapsed_time_build + elapsed_time_search, n, 1);

        free(points);
        free(result);
        free(tree);
        cudaDeviceReset();
    }
    return 0;
}

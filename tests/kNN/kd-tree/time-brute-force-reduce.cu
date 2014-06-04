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
               n_qp = -1;
    n = nu = ni;

    if (argc == 2)
    {
        nu = ni = atoi(argv[1]);
        printf("Running kNN-brute-force with n = %d and k = %d\n", nu , k);

    }
    else if (argc == 3)
    {
        nu = ni = atoi(argv[1]);
        if (atoi(argv[2]))
        {
            k = atoi(argv[2]);
            printf("Running kNN-brute-force with n = %d and k = %d\n", nu , k);
        }
    }
    else if (argc == 4)
    {
        nu = atoi(argv[1]);
        ni = atoi(argv[2]);
        step = atoi(argv[3]);
        printf("Running kNN-brute-force from n = %d to n = %d with step = %d\n", nu, ni, step);
    }
    else if (argc == 5)
    {
        nu = atoi(argv[1]);
        ni = atoi(argv[2]);
        step = atoi(argv[3]);
        k = atoi(argv[4]);
        printf("Running kNN-brute-force from n = %d to n = %d with step = %d and k = %d\n", nu, ni, step, k);
    }
    else if (argc == 6)
    {
        nu = atoi(argv[1]);
        ni = atoi(argv[2]);
        step = atoi(argv[3]);
        k = atoi(argv[4]);
        n_qp = atoi(argv[5]);
        printf("Running kNN-brute-force from n = %d to n = %d with step = %d and k = %d n_qp = %d\n", nu, ni, step, k, n_qp);
    }
    else
    {
        printf("Running kNN-brute-force with n = %d and k = %d\n", nu, k);
    }

    for (n = nu; n <= ni ; n += step)
    {
        float *ref, *dist;
        float *query;
        int *ind;
        unsigned int    query_nb = 1;
        unsigned int    dim = 3;

        ref    = (float *) malloc(n   * dim * sizeof(float));
        query  = (float *) malloc(query_nb * dim * sizeof(float));
        dist  = (float *) malloc(k * sizeof(float));
        ind  = (int *) malloc(k * sizeof(float));

        for (unsigned int count = 0; count < n * dim; count++)
        {
            ref[count] = (float)n * dim - count;
        }

        cudaEvent_t start, stop;
        checkCudaErrors(cudaEventCreate(&start));
        checkCudaErrors(cudaEventCreate(&stop));
        float elapsed_time = 0;

        checkCudaErrors(cudaEventRecord(start, 0));
        if (n_qp <= 0)
        {
            n_qp = 1;
        }
        for (int i = 0; i < n_qp; ++i)
        {
            knn_brute_force(ref, n, ref + (i * dim), dim, k, dist, ind);
        }

        checkCudaErrors(cudaEventRecord(stop, 0));
        cudaEventSynchronize(start);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_time, start, stop);
        printf("kNN-brute-force-reduce, Time = %.5f ms, Size = %u Elements, k = %u NumDevsUsed = %d\n",
               elapsed_time, n, k, 1);

        free(dist);
        free(ind);
        free(query);
        free(ref);
        cudaDeviceReset();
        cudaDeviceSynchronize();
    }
    return 0;
}

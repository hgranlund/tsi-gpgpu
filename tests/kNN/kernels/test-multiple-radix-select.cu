#include "multiple-radix-select.cuh"
#include <knn_gpgpu.h>
#include "test-common.cuh"

#define debug 0
#define THREADS_PER_BLOCK 1024U
#define MAX_BLOCK_DIM_SIZE 65535U


void printPoints2(struct Point *l, int n)
{
    int i;
    if (debug)
    {
        // printf("[(%3.1f, %3.1f, %3.1f)", l[0].p[0], l[0].p[1], l[0].p[2]);
        printf("[%3.1f, ", l[0].p[0]);
        for (i = 1; i < n; ++i)
        {
            printf(", %3.1f, ", l[i].p[0]);
            // printf(", (%3.1f, %3.1f, %3.1f)", l[i].p[0], l[i].p[1], l[i].p[2]);
        }
        printf("]\n");
    }
}


int cpu_partition(struct Point *data, int l, int u, int bit)
{
    unsigned int radix = (1 << (31 - bit));
    struct Point *temp = (struct Point *)malloc(((u - l) + 1) * sizeof(Point));
    int pos = 0;
    for (int i = l; i <= u; i++)
    {
        if (((*(int *) & (data[i].p[0]))&radix))
        {
            temp[pos++] = data[i];
        }
    }
    int result = u - pos;
    for (int i = l; i <= u; i++)
    {
        if (!((*(int *) & (data[i]))&radix))
        {
            temp[pos++] = data[i];
        }
    }
    pos = 0;
    for (int i = u; i >= l; i--)
    {
        data[i] = temp[pos++];
    }

    free(temp);
    return result;
}

struct Point cpu_radixselect(struct Point *data, int l, int u, int m, int bit)
{

    struct Point t;
    t.p[0] = 0;
    t.p[1] = 0;
    t.p[2] = 0;
    if (l == u) return (data[l]);
    if (bit > 32)
    {
        printf("cpu_radixselect fail!\n");
        return t;
    }
    int s = cpu_partition(data, l, u, bit);
    if (s >= m) return cpu_radixselect(data, l, s, m, bit + 1);
    return cpu_radixselect(data, s + 1, u, m, bit + 1);
}

TEST(multiple_radix_select, correctness)
{
    struct Point *h_points, *d_points, *d_swap;
    int n, p, *d_partition, *h_steps, *d_steps, dim = 0;
    for (n = 10; n <= 8000; n += 1000)
    {
        p = 2;
        h_steps = (int *) malloc(p * 2 * sizeof(int));
        h_steps[0] = 0;
        h_steps[1] = n / p;
        h_steps[2] = n / p + 1;
        h_steps[3] = n;

        h_points = (struct Point *) malloc(n * sizeof(Point));
        readPoints("../tests/data/10000_points.data", n, h_points);
        // readPoints("/home/simenhg/workspace/tsi-gpgpu/tests/data/100_mill_points.data", n, h_points);
        // populatePointSRosetta(h_points, n);
        // printPoints2(h_points, n / 2);


        checkCudaErrors(cudaMalloc((void **)&d_points, n  * sizeof(Point)));
        checkCudaErrors(cudaMalloc((void **)&d_swap, n  * sizeof(Point)));
        checkCudaErrors(cudaMalloc((void **)&d_partition, n  * sizeof(int)));
        checkCudaErrors(cudaMalloc((void **)&d_steps, p * 2 * sizeof(int)));

        checkCudaErrors(cudaMemcpy(d_steps, h_steps, p * 2 * sizeof(int), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_points, h_points, n  * sizeof(Point), cudaMemcpyHostToDevice));

        multiRadixSelectAndPartition(d_points, d_swap, d_partition, d_steps, n, p, dim);

        checkCudaErrors(cudaMemcpy(h_points, d_points, n  * sizeof(Point), cudaMemcpyDeviceToHost));
        // printPoints2(h_points, n / 2);

        ASSERT_TREE_LEVEL_OK(h_points, h_steps, n, p, dim);

        checkCudaErrors(cudaFree(d_points));
        checkCudaErrors(cudaFree(d_steps));
        checkCudaErrors(cudaFree(d_swap));
        checkCudaErrors(cudaFree(d_partition));
        free(h_points);
        free(h_steps);
        cudaDeviceSynchronize();
        cudaDeviceReset();
    }
}

TEST(multiple_radix_select, correctness_dim)
{
    struct Point *h_points, *d_points, *d_swap;
    int n = 1024,
        p = 2,
        dim = 0,
        *d_partition,
        *h_steps,
        *d_steps;

    h_steps = (int *) malloc(p * 2 * sizeof(int));
    h_steps[0] = 0;
    h_steps[1] = n / p;
    h_steps[2] = n / p + 1;
    h_steps[3] = n;

    h_points = (struct Point *) malloc(n * sizeof(Point));
    readPoints("../tests/data/10000_points.data", n, h_points);

    checkCudaErrors(cudaMalloc((void **)&d_points, n  * sizeof(Point)));
    checkCudaErrors(cudaMalloc((void **)&d_swap, n  * sizeof(Point)));
    checkCudaErrors(cudaMalloc((void **)&d_partition, n  * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_steps, p * 2 * sizeof(int)));

    checkCudaErrors(cudaMemcpy(d_steps, h_steps, p * 2 * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_points, h_points, n  * sizeof(Point), cudaMemcpyHostToDevice));

    for (dim = 0; dim < 3; dim++)
    {
        multiRadixSelectAndPartition(d_points, d_swap, d_partition, d_steps, n, p, dim);
        checkCudaErrors(cudaMemcpy(h_points, d_points, n  * sizeof(Point), cudaMemcpyDeviceToHost));

        ASSERT_TREE_LEVEL_OK(h_points, h_steps, n, p, dim);

    }
    checkCudaErrors(cudaFree(d_points));
    checkCudaErrors(cudaFree(d_steps));
    checkCudaErrors(cudaFree(d_swap));
    checkCudaErrors(cudaFree(d_partition));
    free(h_points);
    free(h_steps);
    cudaDeviceSynchronize();
}

TEST(multiple_radix_select, timing)
{
    struct Point *h_points, *d_points, *d_swap;
    int n, p, *d_partition, *h_steps, *d_steps;
    for (n = 8388608; n <= 8388608; n <<= 1)
    {
        p = 2;
        h_steps = (int *) malloc(p * 2 * sizeof(int));
        h_steps[0] = 0;
        h_steps[1] = n / p;
        h_steps[2] = n / p + 1;
        h_steps[3] = n;

        h_points = (struct Point *) malloc(n * sizeof(Point));
        populatePointSs(h_points, n);

        checkCudaErrors(cudaMalloc((void **)&d_points, n  * sizeof(Point)));
        checkCudaErrors(cudaMalloc((void **)&d_swap, n  * sizeof(Point)));
        checkCudaErrors(cudaMalloc((void **)&d_partition, n  * sizeof(int)));
        checkCudaErrors(cudaMalloc((void **)&d_steps, p * 2 * sizeof(int)));

        checkCudaErrors(cudaMemcpy(d_steps, h_steps, p * 2 * sizeof(int), cudaMemcpyHostToDevice));


        float elapsed_time = 0;
        cudaEvent_t start, stop;

        cudaStartTiming(start, stop, elapsed_time);

        checkCudaErrors(cudaMemcpy(d_points, h_points, n  * sizeof(Point), cudaMemcpyHostToDevice));
        multiRadixSelectAndPartition(d_points, d_swap, d_partition, d_steps, n, p, 0);

        cudaStopTiming(start, stop, elapsed_time);

        int bytes = n * (sizeof(float)) ;
        printCudaTiming(elapsed_time, bytes, n);

        checkCudaErrors(cudaFree(d_points));
        checkCudaErrors(cudaFree(d_steps));
        checkCudaErrors(cudaFree(d_swap));
        checkCudaErrors(cudaFree(d_partition));
        free(h_points);
        free(h_steps);
        cudaDeviceReset();
    }
}

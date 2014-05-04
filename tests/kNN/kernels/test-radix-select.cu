// Includes
#include <math.h>

#include <radix-select.cuh>
#include <knn_gpgpu.h>
#include "test-common.cuh"

#define debug 0
#define FILE (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define debugf(fmt, ...) if(debug)printf("%s:%d: " fmt, FILE, __LINE__, __VA_ARGS__);


int cpu_partition1(struct Point *data, int l, int u, int bit)
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

struct Point cpu_radixselect1(struct Point *data, int l, int u, int m, int bit)
{
    struct Point t;
    t.p[0] = 0;
    t.p[1] = 0;
    t.p[2] = 0;
    if (l == u) return (data[l]);
    if (bit > 32)
    {
        // debugf("cpu_radixselect1 fail!\n");
        return t;
    }
    int s = cpu_partition1(data, l, u, bit);
    if (s >= m) return cpu_radixselect1(data, l, s, m, bit + 1);
    return cpu_radixselect1(data, s + 1, u, m, bit + 1);
}

void printPoints1(struct Point *l, int n, int dim)
{
    int i;
    if (debug)
    {
        printf("[(%3.1f, %3.1f, %3.1f)", l[0].p[0], l[0].p[1], l[0].p[2]);
        // printf("[%3.1f, ", l[0].p[dim]);
        for (i = 1; i < n; ++i)
        {
            // printf(", %3.1f, ", l[i].p[dim]);
            printf(", (%3.1f, %3.1f, %3.1f)", l[i].p[0], l[i].p[1], l[i].p[2]);
        }
        printf("]\n");
    }
}

__device__ __host__
unsigned int nextPowerOf22(unsigned int x)
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

__device__ __host__
bool isPowTwo2(unsigned int x)
{
    return ((x & (x - 1)) == 0);
}

__device__ __host__
unsigned int prevPowerOf22(unsigned int n)
{
    if (isPowTwo2(n))
    {
        return n;
    }
    n = nextPowerOf22(n);
    return n >>= 1;

}

TEST(radix_selection, correctness)
{
    struct Point *h_points,
            *d_points, *d_temp;
    int n, dim = 0, *partition;

    for (n = 10; n <= 1000000; n += 100000)
{
        h_points = (struct Point *) malloc(n * sizeof(Point));

        checkCudaErrors(
            cudaMalloc((void **)&d_points, n * sizeof(Point)));
        checkCudaErrors(
            cudaMalloc((void **)&d_temp, n * sizeof(Point)));
        checkCudaErrors(
            cudaMalloc((void **)&partition, n * sizeof(int)));

        if (n > 10000)
        {
            populatePointSRosetta(h_points, n);
        }
        else
        {
            readPoints("../tests/data/10000_points.data", n, h_points);
        }
        // printPoints1(h_points, n, dim);

        checkCudaErrors(
            cudaMemcpy(d_points, h_points, n * sizeof(Point), cudaMemcpyHostToDevice));

        radixSelectAndPartition(d_points, d_temp, partition, n, dim);

        checkCudaErrors(
            cudaMemcpy(h_points, d_points, n * sizeof(Point), cudaMemcpyDeviceToHost));

        struct Point cpu_result = cpu_radixselect1(h_points, 0, n - 1, n / 2, 0);
        debugf("result_gpu = (%3.1f, %3.1f, %3.1f)\n", h_points[n / 2].p[0], h_points[n / 2].p[1], h_points[n / 2].p[2] );
        debugf("result_cpu = (%3.1f, %3.1f, %3.1f)\n", cpu_result.p[0], cpu_result.p[1], cpu_result.p[2] );

        int *h_steps = (int *) malloc( 2 * sizeof(int));
        h_steps[0] = 0;
        h_steps[1] = n;

        ASSERT_TREE_LEVEL_OK(h_points, h_steps, n, 1, dim);
        // printPoints1(h_points, n, dim);

        checkCudaErrors(
            cudaFree(d_points));
        checkCudaErrors(
            cudaFree(d_temp));
        checkCudaErrors(
            cudaFree(partition));
        cudaDeviceSynchronize();
        cudaDeviceReset();
    }
}

TEST(radix_selection, correctness_dim)
{
    struct Point *h_points,
            *d_points, *d_temp;
    int dim = 0, n = 10, *partition, *h_steps;

    h_points = (struct Point *) malloc(n * sizeof(Point));
    h_steps = (int *) malloc( 2 * sizeof(int));

    h_steps[0] = 0;
    h_steps[1] = n;

    checkCudaErrors(
        cudaMalloc((void **)&d_points, n * sizeof(Point)));
    checkCudaErrors(
        cudaMalloc((void **)&d_temp, n * sizeof(Point)));
    checkCudaErrors(
        cudaMalloc((void **)&partition, n * sizeof(int)));

    if (n > 10000)
{
        populatePointSRosetta(h_points, n);
    }
    else
    {
        readPoints("../tests/data/10000_points.data", n, h_points);
    }

    checkCudaErrors(
        cudaMemcpy(d_points, h_points, n * sizeof(Point), cudaMemcpyHostToDevice));

    for (dim = 1; dim < 2; dim++)
    {
        radixSelectAndPartition(d_points, d_temp, partition, n, dim);

        struct Point cpu_result = cpu_radixselect1(h_points, 0, n - 1, n / 2, 0);
        debugf("result_gpu = (%3.1f, %3.1f, %3.1f)\n", h_points[n / 2].p[0], h_points[n / 2].p[1], h_points[n / 2].p[2] );
        debugf("result_cpu = (%3.1f, %3.1f, %3.1f)\n", cpu_result.p[0], cpu_result.p[1], cpu_result.p[2] );

        checkCudaErrors(
            cudaMemcpy(h_points, d_points, n * sizeof(Point), cudaMemcpyDeviceToHost));

        printPoints1(h_points, n, dim);

        ASSERT_TREE_LEVEL_OK(h_points, h_steps, n, 1, dim);
    }
    checkCudaErrors(
        cudaFree(d_points));
    checkCudaErrors(
        cudaFree(d_temp));
    checkCudaErrors(
        cudaFree(partition));
    cudaDeviceSynchronize();
    cudaDeviceReset();
}


TEST(radix_selection, timing)
{
    struct Point *h_points;
    int n;

    for (n = 8388608; n <= 8388608; n <<= 1)
    {
        h_points = (struct Point *) malloc(n * sizeof(Point));

        populatePointSRosetta(h_points, n);
        // readPoints("/home/simenhg/workspace/tsi-gpgpu/tests/data/10000_points.data", n, h_points);

        struct Point *d_points, *d_temp;
        int *partition;

        checkCudaErrors(
            cudaMalloc((void **)&d_points, n * sizeof(Point)));
        checkCudaErrors(
            cudaMalloc((void **)&d_temp, n * sizeof(Point)));
        checkCudaErrors(
            cudaMalloc((void **)&partition, n * sizeof(int)));
        checkCudaErrors(
            cudaMemcpy(d_points, h_points, n * sizeof(Point), cudaMemcpyHostToDevice));

        radixSelectAndPartition(d_points, d_temp, partition, n, 0);

        checkCudaErrors(
            cudaMemcpy(h_points, d_points, n * sizeof(Point), cudaMemcpyDeviceToHost));

        float elapsed_time = 0;
        cudaEvent_t start, stop;
        cudaStartTiming(start, stop, elapsed_time);

        checkCudaErrors(cudaMemcpy(d_points, h_points, n  * sizeof(Point), cudaMemcpyHostToDevice));
        radixSelectAndPartition(d_points, d_temp, partition, n, 0);

        cudaStopTiming(start, stop, elapsed_time);

        int bytes = n * (sizeof(float)) ;
        printCudaTiming(elapsed_time, bytes, n);

        checkCudaErrors(
            cudaMemcpy(h_points, d_points, n * sizeof(Point), cudaMemcpyDeviceToHost));

        checkCudaErrors(
            cudaFree(d_points));
        checkCudaErrors(
            cudaFree(partition));
        checkCudaErrors(
            cudaFree(d_temp));
        cudaDeviceSynchronize();
        cudaDeviceReset();
    }
}



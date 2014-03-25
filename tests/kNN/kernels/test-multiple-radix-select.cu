// Includes
#include "multiple-radix-select.cuh"
#include <knn_gpgpu.h>
#include <stdio.h>
#include <gtest/gtest.h>

#include <math.h>
#include <cuda.h>
#include <time.h>
#include <assert.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#define checkCudaErrors(val)           check ( (val), #val, __FILE__, __LINE__ )
#define inf 0x7f800000
#define THREADS_PER_BLOCK 1024U
#define MAX_BLOCK_DIM_SIZE 65535U
#define debug 0
#define FILE (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define debugf(fmt, ...) if(debug)printf("%s:%d: " fmt, FILE, __LINE__, __VA_ARGS__);


int cpu_partition(Point *data, int l, int u, int bit)
{
    unsigned int radix = (1 << 31 - bit);
    Point *temp = (Point *)malloc(((u - l) + 1) * sizeof(Point));
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

Point cpu_radixselect(Point *data, int l, int u, int m, int bit)
{

    Point t;
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

void printPoints(Point *l, int n)
{
    int i;
    if (debug)
    {
        printf("[%3.1f, ", l[0].p[0]);
        for (i = 1; i < n; ++i)
        {
            printf(", %3.1f, ", l[i].p[0]);
        }
        printf("]\n");
    }
}

__device__ __host__
unsigned int nextPowerOf21(unsigned int x)
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
bool isPowTwo1(unsigned int x)
{
    return ((x & (x - 1)) == 0);
}

__device__ __host__
unsigned int prevPowerOf21(unsigned int n)
{
    if (isPowTwo1(n))
    {
        return n;
    }
    n = nextPowerOf21(n);
    return n >>= 1;
}

TEST(kernels, multi_radix_selection)
{
    Point *h_points;
    int numBlocks, numThreads;
    float temp;
    int i, n;
    for (n = 4; n <= 5000; n <<= 1)
    {
        h_points = (Point *) malloc(n * sizeof(Point));
        srand ( (unsigned int)time(NULL) );
        for (i = 0 ; i < n; i++)
        {
            temp =  (float) i;
            temp =  (float) rand() / 100000000;
            Point t;
            t.p[0] = temp;
            t.p[1] = temp;
            t.p[2] = temp;
            h_points[i]    = t;
        }

        printPoints(h_points, n);

        Point *d_points, *d_temp;
        int *partition;
        checkCudaErrors(
            cudaMalloc((void **)&d_points, n * sizeof(Point)));
        checkCudaErrors(
            cudaMalloc((void **)&d_temp, n * sizeof(Point)));
        checkCudaErrors(
            cudaMalloc((void **)&partition, n * sizeof(int)));
        checkCudaErrors(
            cudaMemcpy(d_points, h_points, n * sizeof(Point), cudaMemcpyHostToDevice));


        Point cpu_result = cpu_radixselect(h_points, 0, n - 1, n / 2, 0);
        getThreadAndBlockCountMulRadix(n, 1, numBlocks, numThreads);
        debugf("threads = %d, n = %d\n", numThreads, n);
        cuRadixSelectGlobal <<< numBlocks, numThreads>>>(d_points, d_temp, n, partition, 0);
        checkCudaErrors(
            cudaMemcpy(h_points, d_points, n * sizeof(Point), cudaMemcpyDeviceToHost));

        printPoints(h_points, n);

        debugf("result = (%3.1f, %3.1f, %3.1f)\n", h_points[n / 2].p[0], h_points[n / 2].p[1], h_points[n / 2].p[2] );
        ASSERT_EQ(cpu_result.p[0], h_points[n / 2].p[0]) << "Faild with n = " << n;
        ASSERT_EQ(cpu_result.p[1], h_points[n / 2].p[1]) << "Faild with n = " << n;
        ASSERT_EQ(cpu_result.p[2], h_points[n / 2].p[2]) << "Faild with n = " << n;

        for (int i = 0; i < n / 2; ++i)
        {
            ASSERT_LE(h_points[i].p[0], h_points[n / 2].p[0]) << "Faild with n = " << n;
        }
        for (int i = n / 2; i < n; ++i)
        {
            ASSERT_GE(h_points[i].p[0], h_points[n / 2].p[0]) << "Faild with n = " << n;
        }
        checkCudaErrors(
            cudaFree(d_points));
        checkCudaErrors(
            cudaFree(partition));
        free(h_points);
        cudaDeviceSynchronize();
        cudaDeviceReset();
    }
}

TEST(kernels, multi_radix_selection_time)
{
    Point *h_points;
    unsigned int i, n;
    n = 160000;
    int numBlocks, numThreads,
        p = 1;
    for (n = 8388608; n <= 8388608; n <<= 1)
    {
        h_points = (Point *) malloc(n * p * sizeof(Point));

        float temp;

        h_points = (Point *) malloc(n * p * sizeof(Point));
        srand ( (unsigned int)time(NULL) );
        for (i = 0 ; i < n * p; i++)
        {
            temp =  (float) rand() / 100000000;
            Point t;
            t.p[0] = temp;
            t.p[1] = temp;
            t.p[2] = temp;
            h_points[i]    = t;
        }

        printPoints(h_points, n);

        Point *d_points, *d_temp;
        int *partition;
        checkCudaErrors(
            cudaMalloc((void **)&d_points, n * p * sizeof(Point)));
        checkCudaErrors(
            cudaMalloc((void **)&d_temp, n * p * sizeof(Point)));
        checkCudaErrors(
            cudaMalloc((void **)&partition, n * p * sizeof(int)));
        checkCudaErrors(
            cudaMemcpy(d_points, h_points, n * p * sizeof(Point), cudaMemcpyHostToDevice));



        cudaEvent_t start, stop;
        unsigned int bytes = n * (sizeof(float)) ;
        checkCudaErrors(cudaEventCreate(&start));
        checkCudaErrors(cudaEventCreate(&stop));
        float elapsed_time = 0;
        checkCudaErrors(cudaEventRecord(start, 0));

        getThreadAndBlockCountMulRadix(n, p, numBlocks, numThreads);
        cuRadixSelectGlobal <<< numBlocks, numThreads>>>(d_points, d_temp, n, partition, 0);

        checkCudaErrors(cudaEventRecord(stop, 0));
        cudaEventSynchronize(start);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_time, start, stop);
        elapsed_time = elapsed_time ;
        double throughput = 1.0e-9 * ((double)bytes) / (elapsed_time * 1e-3);
        printf("multi-radix-select, Throughput = %.4f GB/s, Time = %.5f ms, Size = %u Elements, NumDevsUsed = %d\n",
               throughput, elapsed_time, n, 1);

        checkCudaErrors(
            cudaMemcpy(h_points, d_points, n * sizeof(Point), cudaMemcpyDeviceToHost));

        printPoints(h_points, n);

        checkCudaErrors(
            cudaFree(d_points));
        checkCudaErrors(
            cudaFree(partition));
        checkCudaErrors(
            cudaFree(d_temp));
        free(h_points);
        cudaDeviceReset();
    }
}


// Includes
#include <radix-select.cuh>
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


int cpu_partition1(PointS *data, int l, int u, int bit)
{
    unsigned int radix = (1 << 31 - bit);
    PointS *temp = (PointS *)malloc(((u - l) + 1) * sizeof(PointS));
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

PointS cpu_radixselect1(PointS *data, int l, int u, int m, int bit)
{

    PointS t;
    t.p[0] = 0;
    t.p[1] = 0;
    t.p[2] = 0;
    if (l == u) return (data[l]);
    if (bit > 32)
    {
        printf("cpu_radixselect1 fail!\n");
        return t;
    }
    int s = cpu_partition1(data, l, u, bit);
    if (s >= m) return cpu_radixselect1(data, l, s, m, bit + 1);
    return cpu_radixselect1(data, s + 1, u, m, bit + 1);
}






void printPoints1(PointS *l, int n)
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
    PointS *h_points;
    float temp;
    int i, n;
    for (n = 4; n <= 16384; n <<= 1)
    {
        h_points = (PointS *) malloc(n * sizeof(PointS));
        srand ( (unsigned int)time(NULL) );
        for (i = 0 ; i < n; i++)
        {
            temp =  (float) rand() / 100000000;
            temp =  (float) i;
            PointS t;
            t.p[0] = temp;
            t.p[1] = temp;
            t.p[2] = temp;
            h_points[i]    = t;
        }

        printPoints1(h_points, n);

        PointS *d_points, *d_temp;
        int *partition;
        checkCudaErrors(
            cudaMalloc((void **)&d_points, n * sizeof(PointS)));
        checkCudaErrors(
            cudaMalloc((void **)&d_temp, n * sizeof(PointS)));
        checkCudaErrors(
            cudaMalloc((void **)&partition, n * sizeof(int)));
        checkCudaErrors(
            cudaMemcpy(d_points, h_points, n * sizeof(PointS), cudaMemcpyHostToDevice));


        PointS cpu_result = cpu_radixselect1(h_points, 0, n - 1, n / 2, 0);

        radixSelectAndPartition(d_points, d_temp, partition, n, 0);

        checkCudaErrors(
            cudaMemcpy(h_points, d_points, n * sizeof(PointS), cudaMemcpyDeviceToHost));

        printPoints1(h_points, n);

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
            cudaFree(d_temp));
        checkCudaErrors(
            cudaFree(partition));
        cudaDeviceSynchronize();
        cudaDeviceReset();
    }
}


TEST(radix_selection, timing)
{
    PointS *h_points;
    float temp;
    int i, n;
    for (n = 8388608; n <= 8388608; n <<= 1)
    {
        h_points = (PointS *) malloc(n * sizeof(PointS));
        srand ( (unsigned int)time(NULL) );
        for (i = 0 ; i < n; i++)
        {
            temp =  (float) n - 1 - i;
            temp =  (float) rand() / 100000000;
            PointS t;
            t.p[0] = temp;
            t.p[1] = temp;
            t.p[2] = temp;
            h_points[i]    = t;
        }

        PointS *d_points, *d_temp;
        int *partition;
        checkCudaErrors(
            cudaMalloc((void **)&d_points, n * sizeof(PointS)));
        checkCudaErrors(
            cudaMalloc((void **)&d_temp, n * sizeof(PointS)));
        checkCudaErrors(
            cudaMalloc((void **)&partition, n * sizeof(int)));
        checkCudaErrors(
            cudaMemcpy(d_points, h_points, n * sizeof(PointS), cudaMemcpyHostToDevice));


        cudaEvent_t start, stop;
        unsigned int bytes = n * (sizeof(PointS)) ;
        checkCudaErrors(cudaEventCreate(&start));
        checkCudaErrors(cudaEventCreate(&stop));
        float elapsed_time = 0;

        checkCudaErrors(cudaEventRecord(start, 0));

        radixSelectAndPartition(d_points, d_temp, partition, n, 0);

        checkCudaErrors(cudaEventRecord(stop, 0));
        cudaEventSynchronize(start);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_time, start, stop);
        elapsed_time = elapsed_time ;
        double throughput = 1.0e-9 * ((double)bytes) / (elapsed_time * 1e-3);
        printf("radix-select, Throughput = %.4f GB/s, Time = %.5f ms, Size = %u Elements, NumDevsUsed = %d\n",
               throughput, elapsed_time, n, 1);

        checkCudaErrors(
            cudaMemcpy(h_points, d_points, n * sizeof(PointS), cudaMemcpyDeviceToHost));


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



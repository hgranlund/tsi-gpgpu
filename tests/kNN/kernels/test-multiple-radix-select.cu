// Includes
#include "multiple-radix-select.cuh"
#include <knn_gpgpu.h>
#include "test-common.cuh"


#define THREADS_PER_BLOCK 1024U
#define MAX_BLOCK_DIM_SIZE 65535U
#define debug 0
#define FILE (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define debugf(fmt, ...) if(debug)printf("%s:%d: " fmt, FILE, __LINE__, __VA_ARGS__);


int cpu_partition(struct PointS *data, int l, int u, int bit)
{
    unsigned int radix = (1 << (31 - bit));
    struct PointS *temp = (struct PointS *)malloc(((u - l) + 1) * sizeof(PointS));
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

struct PointS cpu_radixselect(struct PointS *data, int l, int u, int m, int bit)
{

    struct PointS t;
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

void printPoints(struct PointS *l, int n)
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



TEST(multiple_radix_select, correctness)
{
    struct PointS *h_points;
    float temp;
    int i, n, p, *h_steps, *d_steps;
    for (n = 8; n <= 1000; n <<= 1)
    {
        p = 2;
        h_steps = (int *) malloc(p * 2 * sizeof(int));
        h_steps[0] = 0;
        h_steps[1] = n / p;
        h_steps[2] = n / p + 1;
        h_steps[3] = n;
        h_points = (struct PointS *) malloc(n * sizeof(PointS));
        srand ( (unsigned int)time(NULL) );
        for (i = 0 ; i < n ; i++)
        {
            temp =  (float) i;
            temp =  (float) rand() / 100000000;
            struct PointS t;
            t.p[0] = temp;
            t.p[1] = temp;
            t.p[2] = temp;
            h_points[i]    = t;
        }
        printPoints(h_points, n );

        struct PointS *d_points, *d_swap;
        int *d_partition;
        checkCudaErrors(
            cudaMalloc((void **)&d_points, n  * sizeof(PointS)));
        checkCudaErrors(
            cudaMalloc((void **)&d_swap, n  * sizeof(PointS)));
        checkCudaErrors(
            cudaMalloc((void **)&d_partition, n  * sizeof(int)));
        checkCudaErrors(
            cudaMalloc((void **)&d_steps, p * 2 * sizeof(int)));
        checkCudaErrors(
            cudaMemcpy(d_steps, h_steps, p * 2 * sizeof(int), cudaMemcpyHostToDevice));
        checkCudaErrors(
            cudaMemcpy(d_points, h_points, n  * sizeof(PointS), cudaMemcpyHostToDevice));

        multiRadixSelectAndPartition(d_points, d_swap, d_partition, d_steps, n, p, 0);

        checkCudaErrors(
            cudaMemcpy(h_points, d_points, n  * sizeof(PointS), cudaMemcpyDeviceToHost));

        printPoints(h_points, n );


        struct PointS *t_points;
        int nn = n;
        for (int i = 0; i < p; ++i)
        {
            t_points = h_points + h_steps[i * 2];
            nn =  h_steps[i * 2 + 1] - h_steps[i * 2];
            for (int i = 0; i < nn / 2; ++i)
            {
                ASSERT_LE(t_points[i].p[0], t_points[nn / 2].p[0]) << "Faild with n = " << nn << " and p " << p;
            }
            for (int i = n / 2; i < nn; ++i)
            {
                ASSERT_GE(t_points[i].p[0], t_points[nn / 2].p[0]) << "Faild with n = " << nn << " and p " << p;
            }
        }

        checkCudaErrors(
            cudaFree(d_points));
        checkCudaErrors(
            cudaFree(d_steps));
        checkCudaErrors(
            cudaFree(d_swap));
        checkCudaErrors(
            cudaFree(d_partition));
        free(h_points);
        free(h_steps);
        cudaDeviceSynchronize();
        cudaDeviceReset();
    }
}

TEST(multiple_radix_select, timing)
{
    struct PointS *h_points;
    float temp;
    int i, n, p, *h_steps, *d_steps;
    for (n = 8388608; n <= 8388608; n <<= 1)
    {
        p = 2;
        h_steps = (int *) malloc(p * 2 * sizeof(int));
        h_steps[0] = 0;
        h_steps[1] = n / p;
        h_steps[2] = n / p + 1;
        h_steps[3] = n;
        h_points = (struct PointS *) malloc(n * sizeof(PointS));
        srand ( (unsigned int)time(NULL) );
        for (i = 0 ; i < n ; i++)
        {
            temp =  (float) i;
            temp =  (float) rand() / 100000000;
            struct PointS t;
            t.p[0] = temp;
            t.p[1] = temp;
            t.p[2] = temp;
            h_points[i]    = t;
        }
        printPoints(h_points, n );

        struct PointS *d_points, *d_swap;
        int *d_partition;
        checkCudaErrors(
            cudaMalloc((void **)&d_points, n  * sizeof(PointS)));
        checkCudaErrors(
            cudaMalloc((void **)&d_swap, n  * sizeof(PointS)));
        checkCudaErrors(
            cudaMalloc((void **)&d_partition, n  * sizeof(int)));
        checkCudaErrors(
            cudaMalloc((void **)&d_steps, p * 2 * sizeof(int)));
        checkCudaErrors(
            cudaMemcpy(d_steps, h_steps, p * 2 * sizeof(int), cudaMemcpyHostToDevice));

        cudaEvent_t start, stop;
        unsigned int bytes = n * (sizeof(float)) ;
        checkCudaErrors(cudaEventCreate(&start));
        checkCudaErrors(cudaEventCreate(&stop));
        float elapsed_time = 0;
        checkCudaErrors(cudaEventRecord(start, 0));

        checkCudaErrors(
            cudaMemcpy(d_points, h_points, n  * sizeof(PointS), cudaMemcpyHostToDevice));
        multiRadixSelectAndPartition(d_points, d_swap, d_partition, d_steps, n, p, 0);

        checkCudaErrors(cudaEventRecord(stop, 0));
        cudaEventSynchronize(start);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_time, start, stop);
        elapsed_time = elapsed_time ;
        double throughput = 1.0e-9 * ((double)bytes) / (elapsed_time * 1e-3);
        printf("multi-radix-select, Throughput = %.4f GB/s, Time = %.5f ms, Size = %u Elements, NumDevsUsed = %d\n",
               throughput, elapsed_time, n, 1);


        checkCudaErrors(
            cudaFree(d_points));
        checkCudaErrors(
            cudaFree(d_steps));
        checkCudaErrors(
            cudaFree(d_swap));
        checkCudaErrors(
            cudaFree(d_partition));
        free(h_points);
        free(h_steps);
        cudaDeviceSynchronize();
        cudaDeviceReset();
    }
}

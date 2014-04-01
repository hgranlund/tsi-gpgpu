// Includes
#include <quick-select.cuh>
#include <kd-tree-naive.cuh>
#include <knn_gpgpu.h>
#include <stdio.h>
#include <gtest/gtest.h>
#include <math.h>
#include <cuda.h>
#include <time.h>
#include <assert.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#define inf 0x7f800000
#define THREADS_PER_BLOCK 1024U
#define MAX_BLOCK_DIM_SIZE 65535U

#define debug 0

#define checkCudaErrors(val)           check ( (val), #val, __FILE__, __LINE__ )
#define FILE (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define debugf(fmt, ...) if(debug)printf("%s:%d: " fmt, FILE, __LINE__, __VA_ARGS__);



__host__  void h_printPointsArray_(PointS *l, int n, char *s, int l_debug = 0)
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

TEST(kernels, quick_selection)
{
    PointS *h_points, *d_points;
    int  *d_steps, *h_steps;
    float temp;
    unsigned int i, n, p;
    for (n = 8; n <= 5000; n <<= 1)
    {
        p = 2;
        h_steps = (int *) malloc(p * 2 * sizeof(int));
        h_steps[0] = 0;
        h_steps[1] = n / p;
        h_steps[2] = n / p + 1;
        h_steps[3] = n;
        h_points = (PointS *) malloc(n  * sizeof(PointS));
        int step = h_steps[1] - h_steps[0];
        srand ( (unsigned int)time(NULL) );
        for (i = 0 ; i < n ; i++)
        {
            temp =  (float) rand() / 100000000;
            PointS t;
            t.p[0] = temp;
            t.p[1] = temp;
            t.p[2] = temp;
            h_points[i]    = t;
        }

        checkCudaErrors(
            cudaMalloc((void **)&d_points, n  * sizeof(PointS)));
        checkCudaErrors(
            cudaMalloc((void **)&d_steps, p * 2  * sizeof(PointS)));

        checkCudaErrors(
            cudaMemcpy(d_points, h_points, n  * sizeof(PointS), cudaMemcpyHostToDevice));
        checkCudaErrors(
            cudaMemcpy(d_steps, h_steps, p * 2  * sizeof(int), cudaMemcpyHostToDevice));

        h_printPointsArray_(h_points, n , "h_points      ", 0);

        quickSelectAndPartition(d_points, d_steps, step , p, 0);

        checkCudaErrors(
            cudaMemcpy(h_points, d_points, n  * sizeof(PointS), cudaMemcpyDeviceToHost));

        h_printPointsArray_(h_points, n , "h_points after", 0);

        PointS *t_points;
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
        free(h_steps);
        free(h_points);
        cudaDeviceSynchronize();
        cudaDeviceReset();
    }
}
TEST(kernels, quick_selection_time)
{
    PointS *h_points, *d_points;
    int  *d_steps, *h_steps;
    float temp;
    unsigned int i, n, p;
    for (n = 2048; n <= 2048; n <<= 1)
    {
        p = 2;
        h_steps = (int *) malloc(p * 2 * sizeof(int));
        h_steps[0] = 0;
        h_steps[1] = n / p;
        h_steps[2] = n / p + 1;
        h_steps[3] = n;
        h_points = (PointS *) malloc(n  * sizeof(PointS));
        int step = h_steps[1] - h_steps[0];

        srand ( (unsigned int)time(NULL) );
        for (i = 0 ; i < n ; i++)
        {
            temp =  (float) rand() / 100000000;
            PointS t;
            t.p[0] = temp;
            t.p[1] = temp;
            t.p[2] = temp;
            h_points[i]    = t;
        }

        checkCudaErrors(
            cudaMalloc((void **)&d_points, n  * sizeof(PointS)));
        checkCudaErrors(
            cudaMalloc((void **)&d_steps, p * 2  * sizeof(PointS)));

        checkCudaErrors(
            cudaMemcpy(d_points, h_points, n  * sizeof(PointS), cudaMemcpyHostToDevice));

        cudaEvent_t start, stop;
        unsigned int bytes = n * (sizeof(PointS));
        checkCudaErrors(cudaEventCreate(&start));
        checkCudaErrors(cudaEventCreate(&stop));
        float elapsed_time = 0;
        checkCudaErrors(cudaEventRecord(start, 0));

        checkCudaErrors(
            cudaMemcpy(d_steps, h_steps, p * 2  * sizeof(int), cudaMemcpyHostToDevice));
        quickSelectAndPartition(d_points, d_steps, step , p, 0);


        checkCudaErrors(cudaEventRecord(stop, 0));
        cudaEventSynchronize(start);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_time, start, stop);
        elapsed_time = elapsed_time ;
        double throughput = 1.0e-9 * ((double)bytes) / (elapsed_time * 1e-3);
        printf("quick_selection, Throughput = %.4f GB/s, Time = %.5f ms, Size = %d, p = %d, NumDevsUsed = %d\n",
               throughput, elapsed_time, n, p, 1);

        checkCudaErrors(
            cudaMemcpy(h_points, d_points, n  * sizeof(PointS), cudaMemcpyDeviceToHost));

        checkCudaErrors(
            cudaFree(d_points));
        checkCudaErrors(
            cudaFree(d_steps));
        free(h_steps);
        free(h_points);
        cudaDeviceSynchronize();
        cudaDeviceReset();
    }
}


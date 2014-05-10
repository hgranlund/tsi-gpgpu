#include <reduction-mod.cuh>
#include <knn_gpgpu.h>
#include "test-common.cuh"
#include <float.h>

float cpu_min(float *in, int num_els)
{
    float min = FLT_MAX;

    for (int i = 0; i < num_els; i++)
    {
        min = in[i] < min ? in[i] : min;
    }

    return min;
}

TEST(min_reduce, correcness)
{
    int i, n;
    Distance *h_dist, *d_dist;

    for (n = 2; n <= 30000000; n <<= 1)
    {
        h_dist = (Distance *) malloc(n * sizeof(Distance));

        srand((int)time(NULL));
        for (i = 0 ; i < n; i++)
        {
            h_dist[i].value = (float) n - i - 1;
            h_dist[i].index = i;
        }

        cudaMalloc((void **) &d_dist, n * sizeof(Distance));
        cudaMemcpy(d_dist, h_dist, n * sizeof(Distance), cudaMemcpyHostToDevice);

        dist_min_reduce(d_dist, n);

        cudaMemcpy(h_dist, d_dist, n * sizeof(Distance), cudaMemcpyDeviceToHost);

        ASSERT_EQ(h_dist[0].value, 0)  << "Faild with n = " << n;
        ASSERT_EQ(h_dist[0].index, n - 1)  << "Faild with n = " << n;

        cudaFree(d_dist);
        free(h_dist);
        cudaDeviceReset();
    }
}

TEST(min_reduce, timing)
{
    int i, n = 8388608;
    Distance *d_dist,
             *h_dist = (Distance *) malloc(n * sizeof(Distance));

    srand((int)time(NULL));

    for (i = 0; i < n; i++)
    {
        h_dist[i].value = (float) n - i - 1;
        h_dist[i].index = i;
    }

    cudaMalloc( (void **) &d_dist, n * sizeof(Distance));
    cudaMemcpy(d_dist, h_dist, n * sizeof(Distance), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    int  bytes = n * (sizeof(Distance) + sizeof(int));
    float elapsed_time = 0;

    cudaStartTiming(start, stop, elapsed_time);
    dist_min_reduce(d_dist, n);
    cudaStopTiming(start, stop, elapsed_time);

    printCudaTiming(elapsed_time, bytes, n);

    cudaFree(d_dist);
    free(h_dist);
    cudaDeviceReset();
}

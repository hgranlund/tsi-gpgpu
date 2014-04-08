// Includes
// #include <reduction.cuh>
#include <reduction-mod.cuh>
#include <knn_gpgpu.h>
#include "test-common.cuh"

#define inf 0x7f800000
#define debug 0
float cpu_min(float *in, int num_els)
{
    float min = inf;

    for (int i = 0; i < num_els; i++)
        min = in[i] < min ? in[i] : min;

    return min;
}

void printDistArray(Distance *l, int n)
{
    int i;
    if (debug)
    {
        printf("[(%d - %3.1f)", l[0].index, l[0].value );
        for (i = 1; i < n; ++i)
        {
            printf(", (%d - %3.1f)", l[i].index, l[i].value );
        }
        printf("]\n");
    }
}

void printIntArray(int *l, int n)
{
    int i;
    if (debug)
    {
        printf("[%4d", l[0] );
        for (i = 1; i < n; ++i)
        {
            printf(", %4d", l[i] );
        }
        printf("]\n");
    }
}


// TEST(min_reduce, min_reduce){

//   Distance *h_dist;
//   int i,n;
//   for (n = 11; n <=11; n +=2)
//   {

//     h_dist = (Distance*) malloc(n*sizeof(Distance));

//     srand ( (unsigned int)time(NULL) );
//     for (i=0 ; i<n; i++)
//     {
//       h_dist[i].value    = n-i-1;
//       h_dist[i].value=i;
//     }
//     // printf("########\n");
//     // printFloatArray(list,n);
//     // printIntArray(ind_1,n);

//     Distance *d_dist;

//     cudaMalloc( (void **) &d_dist, n* sizeof(Distance));

//     cudaMemcpy(d_dist,h_dist, n*sizeof(Distance), cudaMemcpyHostToDevice);

//     knn_min_reduce(d_dist, n);

//     cudaMemcpy(h_dist,d_dist, n*sizeof(Distance), cudaMemcpyDeviceToHost);

//     ASSERT_LE(h_dist[0].value, 0)  << "Faild with n = " << n;
//     ASSERT_LE(h_dist[0].index, n-1)  << "Faild with n = " << n;

//     cudaFree(d_dist);
//     free(h_dist);
//   }
// }

TEST(min_reduce, correcness)
{
    cudaDeviceReset();

    Distance *h_dist;
    unsigned int i, n;
    for (n = 2; n <= 30000000; n <<= 1)
    {

        h_dist = (Distance *) malloc(n * sizeof(Distance));

        srand ( (unsigned int)time(NULL) );
        for (i = 0 ; i < n; i++)
        {
            h_dist[i].value    = (float) n - i - 1;
            h_dist[i].index = i;
        }
        // printf("########\n");
        // printDistArray(h_dist,n);
        // printIntArray(ind_1,n);

        Distance *d_dist;

        cudaMalloc( (void **) &d_dist, n * sizeof(Distance));

        cudaMemcpy(d_dist, h_dist, n * sizeof(Distance), cudaMemcpyHostToDevice);

        dist_min_reduce(d_dist, n);

        cudaMemcpy(h_dist, d_dist, n * sizeof(Distance), cudaMemcpyDeviceToHost);

        // printDistArray(h_dist,n);

        ASSERT_EQ(h_dist[0].value, 0)  << "Faild with n = " << n;
        ASSERT_EQ(h_dist[0].index, n - 1)  << "Faild with n = " << n;
        cudaFree(d_dist);
        free(h_dist);
        cudaDeviceSynchronize();
        cudaDeviceReset();
    }
}

TEST(min_reduce, timing)
{
    cudaDeviceSynchronize();
    cudaDeviceReset();
    Distance *h_dist;
    Distance *d_dist;
    unsigned int i, n;
    n = 8388608;
    h_dist = (Distance *) malloc(n * sizeof(Distance));

    srand ( (unsigned int)time(NULL) );
    for (i = 0 ; i < n; i++)
    {
        h_dist[i].value    = (float)n - i - 1;
        h_dist[i].index = i;
    }

    cudaMalloc( (void **) &d_dist, n * sizeof(Distance));
    cudaMemcpy(d_dist, h_dist, n * sizeof(Distance), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    unsigned int bytes = n * (sizeof(Distance) + sizeof(int));
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    float elapsed_time = 0;

    checkCudaErrors(cudaEventRecord(start, 0));


    dist_min_reduce(d_dist, n);

    checkCudaErrors(cudaEventRecord(stop, 0));
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    elapsed_time = elapsed_time ;
    double throughput = 1.0e-9 * ((double)bytes) / (elapsed_time * 1e-3);
    printf("Reduction_mod, Throughput = %.4f GB/s, Time = %.5f ms, Size = %u Elements, NumDevsUsed = %d\n",
           throughput, elapsed_time, n, 1);

    cudaMemcpy(h_dist, d_dist, n * sizeof(Distance), cudaMemcpyDeviceToHost);

    ASSERT_LE(h_dist[0].value, 0)  << "Faild with n = " << n;
    ASSERT_LE(h_dist[0].index, n - 1)  << "Faild with n = " << n;

    cudaFree(d_dist);
    free(h_dist);
    cudaDeviceSynchronize();
    cudaDeviceReset();
}

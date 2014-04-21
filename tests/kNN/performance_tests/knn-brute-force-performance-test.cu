

// Includes
#include <kNN-brute-force-bitonic.cuh>
#include <kNN-brute-force-reduce.cuh>
#include <knn_gpgpu.h>
#include <stdio.h>

#include <math.h>
#include <cuda.h>
#include <time.h>
#include <assert.h>
#include "helper_cuda.h"


#define SHARED_SIZE_LIMIT 1024U
#define checkCudaErrors(val)           check ( (val), #val, __FILE__, __LINE__ )

void  run_iteration(int ref_nb, int k, int iterations)
{
    float *ref;
    float *query;
    float *dist;
    int   *ind;
    int    query_nb     = 1;
    int    dim        = 3;
    int    i;
    ref    = (float *) malloc(ref_nb   * dim * sizeof(float));
    query  = (float *) malloc(query_nb * dim * sizeof(float));
    dist   = (float *) malloc( k * sizeof(float));
    ind    = (int *)   malloc( k * sizeof(int));

    srand ( (unsigned int)time(NULL) );
    for (i = 0 ; i < ref_nb   * dim ; i++)
    {
        ref[i]    = (float)rand() / (float)1000;
    }
    for (i = 0 ; i < query_nb * dim ; i++)
    {
        query[i]  = (float)rand() / (float)1000;
    }


    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    float elapsed_time = 0;

    checkCudaErrors(cudaEventRecord(start, 0));

    for (int i = 0; i < iterations; ++i)
    {
        knn_brute_force_bitonic(ref, ref_nb, query, dim, k, dist, ind);

        // knn_brute_force_reduce(ref, ref_nb, query, dim, k, dist, ind);
    }
    checkCudaErrors(cudaEventRecord(stop, 0));
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("%d, %d, %f \n", k, ref_nb, elapsed_time / iterations);

    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));

    free(ind);
    free(dist);
    free(query);
    free(ref);
}

int main(int argc, char const *argv[])
{

    printf("Running Knn-brute-force with no memory optimalisations\n");
    printf("k, n, time(ms) \n");
    for (int i = 10000000; i <= 10000000; i <<= 1)
    {
        cudaDeviceSynchronize();
        cudaDeviceReset();
        run_iteration(i, 1, 5);
    }
}

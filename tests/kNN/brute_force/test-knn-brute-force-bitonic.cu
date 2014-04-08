

// Includes
#include <kNN-brute-force-bitonic.cuh>
#include <knn_gpgpu.h>
#include <stdio.h>
#include "gtest/gtest.h"

#include <math.h>
#include <cuda.h>
#include <time.h>
#include <assert.h>


TEST(knn_brute_force_bitonic, correctness)
{

    float *h_dist, *h_dist_orig, *d_dist;
    int *h_ind, *h_ind_orig, *d_ind;
    int i, n;
    for (n = 16384; n <= 16384; n <<= 1)
    {

        h_dist = (float *) malloc(n * sizeof(float));
        h_dist_orig = (float *) malloc(n * sizeof(float));
        h_ind = (int *) malloc(n * sizeof(int));
        h_ind_orig = (int *) malloc(n * sizeof(int));
        srand ( (unsigned int)time(NULL) );
        for (i = 0 ; i < n; i++)
        {
            h_dist_orig[i]    = (float)rand();
            h_ind_orig[i] = i;
        }

        cudaMalloc( (void **) &d_dist, n * sizeof(float));
        cudaMalloc( (void **) &d_ind, n * sizeof(int));

        cudaMemcpy(d_dist, h_dist_orig, n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_ind, h_ind_orig, n * sizeof(int), cudaMemcpyHostToDevice);

        bitonic_sort(d_dist, d_ind, n, 1);
        cudaMemcpy(h_dist, d_dist, n * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_ind, d_ind , n * sizeof(int), cudaMemcpyDeviceToHost);

        float last_value = h_dist[0];
        for (i = 0; i < n; ++i)
        {
            ASSERT_LE(last_value, h_dist[i]) << "Faild with i = " << i << " and n = " << n;
            last_value = h_dist[i];
        }
        for (i = 0; i < n; ++i)
        {
            ASSERT_LE(h_dist[i], h_dist_orig[h_ind[i]]) << "Faild with i = " << i << " and n = " << n;
        }

        free(h_ind);
        free(h_dist);
        cudaFree(d_dist);
        cudaFree(d_ind);
        cudaDeviceSynchronize();
        cudaDeviceReset();
    }
}



// Includes
#include <kNN-brute-force.cuh>
#include <knn_gpgpu.h>
#include <stdio.h>

#include <math.h>
#include <cuda.h>
#include <time.h>
#include <assert.h>
#include "../../../common/common-debug.c"


void  run_iteration(int ref_nb, int k, int iterations){
  float* ref;                 // Pointer to reference point array
  float* query;               // Pointer to query point array
  float* dist;                // Pointer to distance array
  int*   ind;                 // Pointer to index array
  int    query_nb     = 1;   // Reference point number, max=65535
  int    dim        = 3;     // Dimension of points
  int    i;

  // Memory allocation
  ref    = (float *) malloc(ref_nb   * dim * sizeof(float));
  query  = (float *) malloc(query_nb * dim * sizeof(float));
  dist   = (float *) malloc( k * sizeof(float));
  ind    = (int *)   malloc( k * sizeof(int));

  // Init
  srand(time(NULL));
  for (i=0 ; i<ref_nb   * dim ; i++)
  {
    ref[i]    = (float)rand() / (float)RAND_MAX;
  }
  for (i=0 ; i<query_nb * dim ; i++)
  {
    query[i]  = (float)rand() / (float)RAND_MAX;
  }

  // Variables for duration evaluation
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float elapsed_time=0;

  // Display informations

  // Call kNN search CUDA
  cudaEventRecord(start, 0);
  for (int i = 0; i < iterations; ++i)
  {
    knn_brute_force(ref, ref_nb, query, dim, k, dist, ind);
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(start);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_time, start, stop);
  printf("%d, %d, %f \n", k, ref_nb, elapsed_time/iterations);

  // Destroy cuda event object and free memory
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  free(ind);
  free(dist);
  free(query);
  free(ref);
}

int main(int argc, char const *argv[])
{

  printf("Running Knn-brute-force with no memory optimalisations\n");
  printf("k, n, time(ms) \n");
  for (int i = 10600000; i < 83886080; i+=250000)
  {

    cudaDeviceSynchronize();
    run_iteration(i,10,1);
  }

}

#include "gtest/gtest.h"

// Constants used by the program
#define MAX_PITCH_VALUE_IN_BYTES       262144
#define MAX_TEXTURE_WIDTH_IN_BYTES     65536
#define MAX_TEXTURE_HEIGHT_IN_BYTES    32768
#define MAX_PART_OF_FREE_MEMORY_USED   0.9
#define BLOCK_DIM                      16


#include <knn.h>


// Includes
#include <stdio.h>
#include <math.h>
#include <time.h>


TEST(knn_tests, knn_complete_test){




  // Variables and parameters
  float* ref;                 // Pointer to reference point array
  float* query;               // Pointer to query point array
  float* dist;                // Pointer to distance array
  int*   ind;                 // Pointer to index array
  int    ref_nb     = 1;    // Reference point number, max=65535
  int    query_nb   = 6000;   // Query point number,     max=65535
  int    dim        = 3;      // Dimension of points
  int    k          = 20;     // Nearest neighbors to consider
  int    iterations = 100;
  int    i;

  // Memory allocation
  ref    = (float *) malloc(ref_nb   * dim * sizeof(float));
  query  = (float *) malloc(query_nb * dim * sizeof(float));
  dist   = (float *) malloc(query_nb * k * sizeof(float));
  ind    = (int *)   malloc(query_nb * k * sizeof(float));

  srand(time(NULL));
  for (i=0 ; i<ref_nb   * dim ; i++){
    ref[i]    = (float)rand() / (float)RAND_MAX;
  }
  for (i=0 ; i<query_nb * dim ; i++){
    query[i]  = (float)rand() / (float)RAND_MAX;
  }

  // cudaEvent_t start, stop;
  // cudaEventCreate(&start);
  // cudaEventCreate(&stop);
  // float elapsed_time;

  printf("Number of reference points      : %6d\n", ref_nb  );
  printf("Number of query points          : %6d\n", query_nb);
  printf("Dimension of points             : %4d\n", dim     );
  printf("Number of neighbors to consider : %4d\n", k       );
  printf("Processing kNN search           :"                );

  // cudaEventRecord(start, 0);
  for (i=0; i<iterations; i++)
    knn(ref, ref_nb, query, query_nb, dim, k, dist, ind);
  // cudaEventRecord(stop, 0);
  // cudaEventSynchronize(stop);
  // cudaEventElapsedTime(&elapsed_time, start, stop);
  // printf(" done in %f s for %d iterations (%f s by iteration)\n", elapsed_time/1000, iterations, elapsed_time/(iterations*1000));

  // cudaEventDestroy(start);
  // cudaEventDestroy(stop);
  free(ind);
  free(dist);
  free(query);
  free(ref);
  SUCCEED();



}


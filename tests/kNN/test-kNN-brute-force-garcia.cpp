#include "gtest/gtest.h"


#include <knn_gpgpu.h>


// Includes
#include <stdio.h>
#include <math.h>
#include <time.h>

TEST(knn_tests, test_knn_brute_force_give_rigth_result_with_6553_points){

  float* ref;                 // Pointer to reference point array
  float* query;               // Pointer to query point array
  float* dist;                // Pointer to distance array
  int*   ind;                 // Pointer to index array
  int    ref_nb;              // Reference point number, max=65535
  int    query_nb;            // Query point number,     max=65535
  int    dim;                 // Dimension of points
  int    k          = 20;     // Nearest neighbors to consider
  int    iterations = 1;
  int    i;

  char fileName[] = "data/knn_brute_force_6553_ref_points_1_query_point.data";
  FILE* file = fopen(fileName, "rb");

  fread(&ref_nb, sizeof(int), 1, file);
  fread(&query_nb, sizeof(int), 1, file);
  fread(&dim, sizeof(int), 1, file);

  ref    = (float *) malloc(ref_nb   * dim * sizeof(float));
  query  = (float *) malloc(query_nb * dim * sizeof(float));
  dist   = (float *) malloc(query_nb * k * sizeof(float));
  ind    = (int *)   malloc(query_nb * k * sizeof(float));

  for (int count = 0; count < ref_nb*dim; count++)
  {
    fread(&ref[count], sizeof(float), 1, file);
  }
  for (int count = 0; count < query_nb*dim; count++)
  {
    fread(&query[count], sizeof(float), 1, file);
  }

  fclose(file);

  for (i=0; i<iterations; i++){
    knn_brute_force(ref, ref_nb, query, query_nb, dim, k, dist, ind);
  }

  int correct_ind[] = {3261, 2799, 5752, 1837, 522, 5065, 5410, 1915, 2618, 627, 6095, 305, 3375, 269, 6180, 4963, 2216, 3393, 31, 5061};
  for (int i = 0; i < k; ++i)
  {
    ASSERT_EQ(ind[i], correct_ind[i]);
  }

  free(ind);
  free(dist);
  free(query);
  free(ref);
  SUCCEED();
}

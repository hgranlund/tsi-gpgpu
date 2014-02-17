

// Includes
#include <kNN-brute-force-reduce.cuh>

#include <stdio.h>
#include <cuda.h>

int main(int argc, char const *argv[])
{

  float* ref;
  float* query;
  float* dist;
  int*   ind;
  int    ref_nb = 131072;
  int    query_nb = 1;
  int    dim=3;
  int    k          = 50;
  int    iterations = 1;
  int    i;
  bool pass = true;

  ref    = (float *) malloc(ref_nb   * dim * sizeof(float));
  query  = (float *) malloc(query_nb * dim * sizeof(float));
  dist   = (float *) malloc(query_nb * k * sizeof(float));
  ind    = (int *)   malloc(query_nb * k * sizeof(int));
  for (int count = 0; count < ref_nb*dim; count++)
  {
    ref[count] = ref_nb*dim-count;
  }
  for (int count = 0; count < query_nb*dim; count++)
  {
    query[count] = 0;
  }

  for (i=0; i<iterations; i++){
    knn_brute_force_reduce(ref, ref_nb, query, dim, k, dist, ind);
  }

  for (int i = 0; i < k; ++i)
  {
    if (ind[i] != ref_nb-1-i){
      pass=false;
    }
  }
  if (pass)
  {
    printf("PASS: TRUE\n");
  }else {
    printf("pass: FALSE \n");
    
  }
  free(ind);
  free(dist);
  free(query);
  free(ref);
}


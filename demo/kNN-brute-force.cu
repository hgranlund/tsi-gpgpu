// Includes
#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include <time.h>

__constant__  float query_dev[3];

__global__ void cuComputeDistanceGlobal( float* ref, int ref_nb , int dim,  float* dist, int* ind){

  // restiction:
  // dimention=3
  // one query point
  float dx,dy,dz;

  int index = blockIdx.x * dim;
  while (index < ref_nb){
    dx=ref[index] - query_dev[0];
    dy=ref[index + 1] - query_dev[1];
    dz=ref[index + 2] - query_dev[2];
    dist[index/dim] = (dx*dx)+(dy*dy)+(dz*dz);
    ind[index/dim] = index/dim;
    index += gridDim.x * dim;
  }
}


__global__ void cuParallelSqrt(float *dist, int k){
  unsigned int xIndex = blockIdx.x;
  if (xIndex < k){
    dist[xIndex] = sqrt(dist[xIndex]);
  }
}



void knn_brute_force(float* ref_host, int ref_nb, float* query_host, int dim, int k, float* dist_host, int* ind_host){

  unsigned int size_of_float = sizeof(float);
  unsigned int size_of_int   = sizeof(int);

  float        *ref_dev;
  float        *dist_dev;
  int          *ind_dev;

  cudaMalloc( (void **) &dist_dev, ref_nb * size_of_float);
  cudaMalloc( (void **) &ind_dev, ref_nb * size_of_int);
  cudaMalloc( (void **) &ref_dev, ref_nb * size_of_float * dim);


  cudaMemcpy(ref_dev, ref_host, ref_nb*dim*size_of_float, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(query_dev, query_host, dim*size_of_float);

  cuComputeDistanceGlobal<<<256,1>>>(ref_dev, ref_nb, dim, dist_dev, ind_dev);

  // TODO: Sort dev and ind;

  // cuParallelSqrt<<<k,1>>>(dist_dev, k);
  cudaMemcpy(dist_host, dist_dev, k*size_of_float, cudaMemcpyDeviceToHost);
  cudaMemcpy(ind_host,  ind_dev,  k*size_of_int, cudaMemcpyDeviceToHost);


  cudaFree(ref_dev);
  cudaFree(ind_dev);
  cudaFree(query_dev);
}



int main(int argc, char const *argv[])
{

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
  ind    = (int *)   malloc(query_nb * k * sizeof(int));

  for (int count = 0; count < ref_nb*dim; count++)
  {
    fread(&ref[count], sizeof(float), 1, file);
  }
  for (int count = 0; count < query_nb*dim; count++)
  {
    fread(&query[count], sizeof(float), 1, file);
  }

  fclose(file);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float elapsed_time;
  cudaEventRecord(start, 0);
    // Display informations
   printf("Number of reference points      : %6d\n", ref_nb  );
   printf("Dimension of points             : %4d\n", dim     );
   printf("Number of neighbors to consider : %4d\n", k       );
   printf("Processing kNN search           :"                );
  for (i=0; i<iterations; i++){
    knn_brute_force(ref, ref_nb, query, dim, k, dist, ind);
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_time, start, stop);
  printf(" done in %f s \n", elapsed_time/1000);

  printf("\nFound indexes: [");
  for (int i = 0; i < k; ++i)
  {
    printf("%d, ",ind[i] );
  }
  printf("]\n");


  printf("Distances: [");
  for (int i = 0; i < k; ++i)
  {
    printf("%f, ",dist[i]);
  }
  printf("]\n");

  // Destroy cuda event object and free memory
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  int correct_ind[] = {3261, 2799, 5752, 1837, 522, 5065, 5410, 1915, 2618, 627, 6095, 305, 3375, 269, 6180, 4963, 2216, 3393, 31, 5061};
  int pass = 1;
  for (int i = 0; i < k; ++i)
  {
    if(ind[i] != correct_ind[i]){
      pass=0;
    }
  }

  if (pass == 1)
  {
    printf("PASS: True\n");
  }
  else{
    printf("PASS: False\n");

  }

  free(ind);
  free(dist);
  free(query);
  free(ref);

  return 0;
}
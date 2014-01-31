#ifndef _KNN_BRUTE_FORCE_
#define _KNN_BRUTE_FORCE_

__global__ void cuComputeDistanceGlobal( float* ref, int ref_nb , float* query, int dim,  float* dist);

// __global__ void cuInsertionSort(float *dist, , int *ind, int width, int dim, int k){

__global__ void cuParallelSqrt(float *dist, int k);


void knn_brute_force(float* ref_host, int ref_nb, float* query_host, int dim, int k, float* dist_host, int* ind_host);


#endif

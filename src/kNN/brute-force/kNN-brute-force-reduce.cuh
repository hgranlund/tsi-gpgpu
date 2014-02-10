#ifndef _KNN_BRUTE_FORCE_REDUCE_
#define _KNN_BRUTE_FORCE_REDUCE_

__global__ void cuComputeDistance( float* ref, int ref_nb , float* query, int dim,  float* dist);
__global__ void cuParallelSqrt(float *dist, int k);
void min_reduce(float* dist_dev, int* ind_dev, int n, int k, int dir);

void knn_brute_force_reduce(float* ref_host, int ref_nb, float* query_host, int dim, int k, float* dist_host, int* ind_host);


#endif

#ifndef _KNN_BRUTE_FORCE_REDUCE_
#define _KNN_BRUTE_FORCE_REDUCE_
#include <data_types.h>

__global__ void cuComputeDistance( float *ref, unsigned int ref_nb , float *query, unsigned int dim,  Distance *dist);
__global__ void cuParallelSqrt(Distance *dist, unsigned int k);
void min_reduce(Distance *d_dist, unsigned int n, unsigned int k, unsigned int dir);

void knn_brute_force(float *ref_host, int ref_nb, float *query_host, int dim, int k, float *dist_host, int *ind_host);


#endif

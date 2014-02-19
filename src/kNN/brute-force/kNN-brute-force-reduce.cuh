#ifndef _KNN_BRUTE_FORCE_REDUCE_
#define _KNN_BRUTE_FORCE_REDUCE_

struct Distance {
   float value;
   int index;
};



__global__ void cuComputeDistance( float* ref, int ref_nb , float* query, int dim,  Distance* dist);
__global__ void cuParallelSqrt(Distance *dist, int k);
void min_reduce(Distance* d_dist, int n, int k, int dir);

void knn_brute_force_reduce(float* ref_host, int ref_nb, float* query_host, int dim, int k, Distance* h_dist);


#endif

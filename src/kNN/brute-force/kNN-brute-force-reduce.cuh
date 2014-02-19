#ifndef _KNN_BRUTE_FORCE_REDUCE_
#define _KNN_BRUTE_FORCE_REDUCE_

struct Distance {
   float value;
   unsigned int index;
};



__global__ void cuComputeDistance( float* ref, unsigned int ref_nb , float* query, unsigned int dim,  Distance* dist);
__global__ void cuParallelSqrt(Distance *dist, unsigned int k);
void min_reduce(Distance* d_dist, unsigned int n, unsigned int k, unsigned int dir);

void knn_brute_force_reduce(float* ref_host, unsigned int ref_nb, float* query_host, unsigned int dim, unsigned int k, Distance* h_dist);


#endif

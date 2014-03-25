#ifndef _REDUCTION_
#define _REDUCTION_


void min_reduce(Distance *dist, unsigned int n);
__global__ void min_reduction(Distance *dist, unsigned int n, unsigned int threadOffset);
void knn_min_reduce(Distance *dist_dev, unsigned int n);

#endif


#ifndef _REDUCTION_
#define _REDUCTION_


struct Distance {
 float value;
 int index;
};

 void min_reduce(Distance *dist, int n);
__global__ void min_reduction(Distance *dist, int n, int threadOffset);
void knn_min_reduce(Distance* dist_dev, int n);

#endif


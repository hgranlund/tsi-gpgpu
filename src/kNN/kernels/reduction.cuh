// Reduction (min/max/avr/sum), works for any blockDim.x:

#ifndef _REDUCTION_
#define _REDUCTION_

 void min_reduce(float *list, int *ind, int n);
__global__ void min_reduction(float *list,int *ind, int n, int threadOffset);
void knn_min_reduce(float* dist_dev, int* ind_dev, int n);

#endif


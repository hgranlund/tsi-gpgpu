#ifndef _KNN_BRUTE_FORCE_
#define _KNN_BRUTE_FORCE_

__global__ void cuBitonicSort(float *dist, int *ind, int n, int dir);

void bitonic_sort(float *dist_dev, int *ind_dev, int n, int dir);

void knn_brute_force_bitonic(float *ref_host, int ref_nb, float *query_host, int dim, int k, float *dist_host, int *ind_host);


#endif

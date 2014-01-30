#ifndef _KERNELS_H_
#define _KERNELS_H_



void knn_brute_force(float* ref_host, int ref_width, float* query_host, int query_width, int height, int k, float* dist_host, int* ind_host);


#endif //  _KERNELS_H_

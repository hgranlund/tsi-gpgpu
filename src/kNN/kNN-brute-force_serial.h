#ifndef _KNN_BRUTE_FORCE_
#define _KNN_BRUTE_FORCE_


__global__ void cuComputeDistanceTexture(int wA, float * B, int wB, int pB, int dim, float* AB);

__global__ void cuComputeDistanceGlobal( float* A, int wA, int pA, float* B, int wB, int pB, int dim,  float* AB);

__global__ void cuInsertionSort(float *dist, int dist_pitch, int *ind, int ind_pitch, int width, int height, int k);

__global__ void cuParallelSqrt(float *dist, int width, int pitch, int k);

void printErrorMessage(cudaError_t error, int memorySize);

void knn_serial(float* ref_host, int ref_width, float* query_host, int query_width, int height, int k, float* dist_host, int* ind_host);

#endif

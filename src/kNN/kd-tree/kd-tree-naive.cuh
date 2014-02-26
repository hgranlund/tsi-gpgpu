#ifndef _KD_TREE_NAIVE_
#define _KD_TREE_NAIVE_

void build_kd_tree(float *points, int n);
int midpoint(int lower, int upper);
void swap(float *points, int a, int b);
float quick_select(int k, float *points, int lower, int upper, int dim);
void center_median(float *points, int lower, int upper, int dim);

__device__ __host__
int index(int i, int j, int n);

#endif

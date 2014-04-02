#ifndef _KD_SEARCH_
#define _KD_SEARCH_
#include "point.h"

#define THREADS_PER_BLOCK_SEARCH 64U
#define MAX_BLOCK_DIM_SIZE 65535U

int store_locations(Point *tree, int lower, int upper, int n);
void queryAll(Point *h_query_points, Point *tree, int qp_n, int tree_n, int k, int *result);

__device__
int nn(float *qp, Point *tree, int dim, int index);

__global__
void nearest(float *qp, Point *tree, int dim, int index);

#endif

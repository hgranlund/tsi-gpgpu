#ifndef _KD_SEARCH_
#define _KD_SEARCH_
#include "point.h"

int store_locations(Point *tree, int lower, int upper, int n);
void all_nearest(Point *h_query_points, Point *tree, int qp_n, int tree_n);

__device__
int nn(float *qp, Point *tree, int dim, int index);

__global__
void nearest(float *qp, Point *tree, int dim, int index);

#endif
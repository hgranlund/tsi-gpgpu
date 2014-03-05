#ifndef _KD_TREE_NAIVE_
#define _KD_TREE_NAIVE_
#include "point.h"


void build_kd_tree(Point *points, int n);


__device__
void cuRadixSelect(Point *data, Point *data_copy, unsigned int m, unsigned int n, int *partition, int dir, Point *result);

__global__
void cuRadixSelectGlobal(Point *data, Point *data_copy, unsigned int m, unsigned int n, int *partition, int dir, Point *result);

#endif

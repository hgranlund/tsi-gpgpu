#ifndef _KD_TREE_NAIVE_
#define _KD_TREE_NAIVE_
#include "point.h"


void build_kd_tree(Point *points, int n);
void getThreadAndBlockCountMulRadix(int n, int p, int &blocks, int &threads);
void getThreadAndBlockCountForQuickSelect(int n, int p, int &blocks, int &threads);

__global__
void cuBalanceBranch(Point* points, Point* swap, int *partition, int n, int p, int dir);


__device__
void cuRadixSelect(Point *data, Point *data_copy, unsigned int m, unsigned int n, int *partition, int dir);

__global__
void cuRadixSelectGlobal(Point *data, Point *data_copy, unsigned int m, unsigned int n, int *partition, int dir);

__global__
void cuQuickSelectGlobal(Point* points, int n, int p, int dir);

template <int maxStep> __global__
void cuQuickSelectShared(Point* points, int n, int p, int dir);

#endif

#ifndef _KD_TREE_NAIVE_
#define _KD_TREE_NAIVE_
#include "point.h"

void build_kd_tree(PointS *points, int n, Point *points_out);
void getThreadAndBlockCountMulRadix(int n, int p, int &blocks, int &threads);
void getThreadAndBlockCountForQuickSelect(int n, int p, int &blocks, int &threads);

__global__
void cuBalanceBranch(PointS *points, PointS *swap, int *partition, int n, int p, int dir);

__global__
void cuQuickSelectGlobal(PointS *points, int n, int p, int dir);

template <int maxStep> __global__
void cuQuickSelectShared(PointS *points, int n, int p, int dir);

#endif

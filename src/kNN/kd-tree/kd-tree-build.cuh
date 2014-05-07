#ifndef _KD_TREE_NAIVE_
#define _KD_TREE_NAIVE_
#include <point.h>
#include <stack.h>

void build_kd_tree(struct Point *points, int n, struct Node *points_out);
void getThreadAndBlockCountMulRadix(int n, int p, int &blocks, int &threads);
void getThreadAndBlockCountForQuickSelect(int n, int p, int &blocks, int &threads);

__global__
void cuBalanceBranch(struct Point *points, struct Point *swap, int *partition, int n, int p, int dir);

__global__
void cuQuickSelectGlobal(struct Point *points, int n, int p, int dir);

template <int maxStep> __global__
void cuQuickSelectShared(struct Point *points, int n, int p, int dir);

#endif

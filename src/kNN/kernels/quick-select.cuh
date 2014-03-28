#ifndef _QUICK_SELECT_
#define _QUICK_SELECT_
#include "point.h"

#define MAX_SHARED_MEM 49152U
#define THREADS_PER_BLOCK_QUICK 128U
#define MAX_BLOCK_DIM_SIZE 65535U

void quickSelectAndPartition(Point *points, int n, int p, int dir);
void quickSelectShared(Point *points, int n, int p, int dir, int size, int numBlocks, int numThreads);
void getThreadAndBlockCountForQuickSelect(int n, int p, int &blocks, int &threads);

template <int maxStep>
__global__
void cuQuickSelectShared(Point *points, int n, int p, int dir);

__global__
void cuQuickSelectGlobal(Point *points, int n, int p, int dir);

#endif


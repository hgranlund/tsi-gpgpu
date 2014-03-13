#ifndef _QUICK_SELECT_
#define _QUICK_SELECT_
#include "point.h"


void quickSelectShared(Point* points, int n, int p, int dir, int size, int numBlocks, int numThreads);
void getThreadAndBlockCountForQuickSelect(int n, int p, int &blocks, int &threads);

template <int maxStep>
__global__
void cuQuickSelectShared(Point* points, int n, int p, int dir);

__global__
void cuQuickSelectGlobal(Point* points, int n, int p, int dir);

#endif


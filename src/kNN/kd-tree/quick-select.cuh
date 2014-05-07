#ifndef _QUICK_SELECT_
#define _QUICK_SELECT_
#include <point.h>
#include <stack.h>

#define MAX_SHARED_MEM 49152U
#define THREADS_PER_BLOCK_QUICK 64U
#define MAX_BLOCK_DIM_SIZE 65535U

void quickSelectAndPartition(struct Point *d_points, int *d_steps, int n, int p, int dir);
void quickSelectShared(struct Point *points, int *steps, int p, int dir, int size, int numBlocks, int numThreads);
void getThreadAndBlockCountForQuickSelect(int n, int p, int &blocks, int &threads);


#endif


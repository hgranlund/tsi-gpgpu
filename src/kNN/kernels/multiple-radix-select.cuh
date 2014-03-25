#ifndef MULTI_RADIX_SELECT_
#define MULTI_RADIX_SELECT_
#include "point.h"


#define THREADS_PER_BLOCK 1024U
#define MAX_BLOCK_DIM_SIZE 65535U

void getThreadAndBlockCountMulRadix(int n, int p, int &blocks, int &threads);
void  multiRadixSelectAndPartition(Point *data, Point *data_copy, int *partition, int n, int p,  int dir);


__global__
void cuBalanceBranch(Point *points, Point *swap, int *partition, int n, int p, int dir);

__global__
void cuRadixSelectGlobal(Point *data, Point *data_copy, int n, int *partition, int dir);


#endif


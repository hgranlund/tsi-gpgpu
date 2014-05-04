#ifndef MULTI_RADIX_SELECT_
#define MULTI_RADIX_SELECT_
#include "point.h"


#define THREADS_PER_BLOCK_MULTI_RADIX 512U
#define MAX_BLOCK_DIM_SIZE_MULTI_RADIX 65535U

void getThreadAndBlockCountMulRadix(int n, int p, int &blocks, int &threads);
void  multiRadixSelectAndPartition(struct Point *data, struct Point *data_copy, int *partition, int *steps, int n, int p,  int dir);

__global__
void cuBalanceBranch(struct Point *points, struct Point *swap, int *partition, int *steps, int p, int dir);


#endif


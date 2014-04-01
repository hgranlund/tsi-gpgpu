#ifndef _RADIX_SELECT_
#define _RADIX_SELECT_
#include "point.h"


#define THREADS_PER_BLOCK 1024U
#define MAX_BLOCK_DIM_SIZE 65535U
#define MAX_SHARED_MEM 49152U

void radixSelectAndPartition(PointS *points, PointS *swap, int *partition, int n, int dir);

#endif


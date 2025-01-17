#ifndef _RADIX_SELECT_
#define _RADIX_SELECT_
#include <point.h>
#include <stack.h>


#define THREADS_PER_BLOCK_RADIX 512U
#define MAX_BLOCK_DIM_SIZE_RADIX 65535U
#define MAX_SHARED_MEM 49152U

void radixSelectAndPartition(struct Point *points, struct Point *swap, int *partition, int n, int dir);

#endif


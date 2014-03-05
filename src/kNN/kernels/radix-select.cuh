#ifndef _RADIX_SELECT_
#define _RADIX_SELECT_
#include "point.h"


__device__
void cuRadixSelect(Point *data, Point *data_copy, unsigned int m, unsigned int n, int *partition, int dir, Point *result);

Point cpu_radixselect(Point *data, int l, int u, int m, int bit);

#endif


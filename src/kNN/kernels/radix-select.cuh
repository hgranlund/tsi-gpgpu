#ifndef _RADIX_SELECT_
#define _RADIX_SELECT_

__global__
void cuRadixSelect(float *data, unsigned int m, unsigned int n, float *ones, float *zeros, float *result);

float cpu_radixselect(float *data, int l, int u, int m, int bit);

#endif


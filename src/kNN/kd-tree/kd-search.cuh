#ifndef _KD_SEARCH_
#define _KD_SEARCH_
#include "point.h"

#define THREADS_PER_BLOCK_SEARCH 32U
#define MAX_BLOCK_DIM_SIZE 65535U

struct KPoint
{
    int index;
    float dist;
};

int store_locations(struct Point *tree, int lower, int upper, int n);
void queryAll(struct Point *h_query_points, struct Point *tree, int qp_n, int tree_n, int k, int *result);

#endif

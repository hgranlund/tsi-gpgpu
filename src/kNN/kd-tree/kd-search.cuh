#ifndef _KD_SEARCH_
#define _KD_SEARCH_
#include "cu-kd-search.cuh"

#define MIN_NUM_QUERY_POINTS 100U

void queryAll(struct Point *h_query_points, struct Node *tree, int qp_n, int tree_n, int k, int *result, int switch_limit);

#endif

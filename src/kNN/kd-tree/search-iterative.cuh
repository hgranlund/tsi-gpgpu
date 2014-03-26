#ifndef _SEARCH_ITERATIVE_
#define _SEARCH_ITERATIVE_
#include <point.h>

int cashe_indexes(Point *tree, int lower, int upper, int n);
int dfs(Point *tree, int n);
int query_k(float *qp, Point *tree, int dim, int index);

#endif
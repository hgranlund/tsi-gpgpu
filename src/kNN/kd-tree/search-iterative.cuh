#ifndef _SEARCH_ITERATIVE_
#define _SEARCH_ITERATIVE_
#include <point.h>

int cashe_indexes(Point *tree, int lower, int upper, int n);
int dfs(Point *tree, int n);
void push(int *stack, int *pop, int value);
int pop(int *stack, int *eos);
int peek(int *stack, int eos);
int find(int *stack, int eos, int value);
void upDim(int *dim);
void downDim(int *dim);
int query_a(Point *qp, Point *tree, int n);
int query_k(float *qp, Point *tree, int dim, int index);

#endif
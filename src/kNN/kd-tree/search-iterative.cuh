#ifndef _SEARCH_ITERATIVE_
#define _SEARCH_ITERATIVE_
#include <point.h>

void push(int *stack, int *pop, int value);
int pop(int *stack, int *eos);
int peek(int *stack, int eos);
int find(int *stack, int eos, int value);
void upDim(int *dim);
void downDim(int *dim);
int query_a(float *qp, struct Point *tree, int n);

#endif

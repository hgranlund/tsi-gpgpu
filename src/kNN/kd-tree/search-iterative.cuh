#ifndef _SEARCH_ITERATIVE_
#define _SEARCH_ITERATIVE_
#include <point.h>

void push(int **stackPtr, int value);
int pop(int **stackPtr);
int peek(int *stackPtr);
bool isEmpty(int *stackPtr);
void initStack(int *stack, int **stackPtr);
void upDim(int *dim);
void downDim(int *dim);
int query_a(struct Point qp, struct Point *tree, int n);
#endif

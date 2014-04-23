#ifndef _SEARCH_ITERATIVE_
#define _SEARCH_ITERATIVE_
#include <point.h>

struct KPoint
{
    int index;
    float dist;
};

void push(int **stackPtr, int value);
int pop(int **stackPtr);
int peek(int *stackPtr);
bool isEmpty(int *stackPtr);
void initStack(int *stack, int **stackPtr);
void upDim(int *dim);
void downDim(int *dim);
int query_a(struct Point qp, struct Point *tree, int n);
void initKStack(struct KPoint **stack, int n);
void insert(struct KPoint *stack, struct KPoint value, int n);
struct KPoint look(struct KPoint *kStack, int n);
#endif

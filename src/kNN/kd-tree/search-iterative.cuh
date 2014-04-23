#ifndef _SEARCH_ITERATIVE_
#define _SEARCH_ITERATIVE_
#include <point.h>

struct KPoint
{
    int index;
    float dist;
};

void push(int **stack, int value);
int pop(int **stack);
int peek(int *stack);
bool isEmpty(int *stack);
void initStack(int *stack_init, int **stack);

void upDim(int *dim);

void initKStack(struct KPoint **k_stack, int n);
void insert(struct KPoint *k_stack, struct KPoint value, int n);
struct KPoint look(struct KPoint *k_stack, int n);

void kNN(struct Point qp, struct Point *tree, int n, int k, int *result);
#endif

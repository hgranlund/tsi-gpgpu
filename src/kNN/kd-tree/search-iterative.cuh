#ifndef _SEARCH_ITERATIVE_
#define _SEARCH_ITERATIVE_
#include <point.h>

void push(struct SPoint **stack, struct SPoint value);
struct SPoint pop(struct SPoint **stack);
struct SPoint peek(struct SPoint *stack);
bool isEmpty(struct SPoint *stack);
void initStack(struct SPoint **stack);

void upDim(int *dim);

void initKStack(struct KPoint **k_stack, int n);
void insert(struct KPoint *k_stack, struct KPoint value, int n);
struct KPoint look(struct KPoint *k_stack, int n);

void kNN(struct Node qp, struct Node *tree, int n, int k, int *result, int *visited);
void kNN(struct Node qp, struct Node *tree, int n, int k, int *result, int *visited, struct SPoint *stack_ptr, struct KPoint *k_stack_ptr);

void timingDetails();
#endif

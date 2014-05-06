#ifndef _KD_SEARCH_OPENMP
#define _KD_SEARCH_OPENMP
#include <point.h>


void push(struct SPoint **stack, struct SPoint value);
struct SPoint pop(struct SPoint **stack);
struct SPoint peek(struct SPoint *stack);
int isEmpty(struct SPoint *stack);
void initStack(struct SPoint **stack);

void upDim(int *dim);

void initKStack(struct KPoint **k_stack, int n);
void insert(struct KPoint *k_stack, struct KPoint value, int n);
struct KPoint look(struct KPoint *k_stack, int n);

void mpQueryAll(struct Point *query_points, struct Node *tree, int n_qp, int n_tree, int k, int *result);
void kNN(struct Point qp, struct Node *tree, int n, int k, int *result, struct SPoint *stack_ptr, struct KPoint *k_stack_ptr);

#endif

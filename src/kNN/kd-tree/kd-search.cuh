#ifndef _KD_SEARCH_
#define _KD_SEARCH_
#include "point.h"

#define THREADS_PER_BLOCK_SEARCH 128U
#define MAX_BLOCK_DIM_SIZE 65535U

struct KPoint
{
    int index;
    float dist;
};

struct SPoint
{
    int index;
    int dim;
};

void initStack(struct SPoint **stack);
bool isEmpty(struct SPoint *stack);
void push(struct SPoint **stack, struct SPoint value);
struct SPoint pop(struct SPoint **stack);
struct SPoint peek(struct SPoint *stack);

void initKStack(KPoint **k_stack, int n);
void insert(struct KPoint *k_stack, struct KPoint k_point, int n);
struct KPoint look(struct KPoint *k_stack, int n);

void upDim(int *dim);

void kNN(struct Point qp, struct Point *tree, int n, int k, int *result,
         struct SPoint *stack_ptr, struct KPoint *k_stack_ptr);
void queryAll(struct Point *h_query_points, struct Point *tree, int qp_n, int tree_n, int k, int *result);

#endif

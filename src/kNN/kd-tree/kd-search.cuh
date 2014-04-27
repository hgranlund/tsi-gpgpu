#ifndef _KD_SEARCH_
#define _KD_SEARCH_
#include "point.h"

#define THREADS_PER_BLOCK_SEARCH 128U
#define MAX_BLOCK_DIM_SIZE 65535U

__device__ __host__
void initStack(struct SPoint **stack);
__device__ __host__
bool isEmpty(struct SPoint *stack);
__device__ __host__
void push(struct SPoint **stack, struct SPoint value);
__device__ __host__
struct SPoint pop(struct SPoint **stack);
__device__ __host__
struct SPoint peek(struct SPoint *stack);
__device__ __host__
void initKStack(KPoint **k_stack, int n);
__device__ __host__
void insert(struct KPoint *k_stack, struct KPoint k_point, int n);
__device__ __host__
struct KPoint look(struct KPoint *k_stack, int n);

__device__ __host__
void upDim(int *dim);

__device__ __host__
void kNN(struct Point qp, struct Point *tree, int n, int k, int *result,
         struct SPoint *stack_ptr, struct KPoint *k_stack_ptr);
void queryAll(struct Point *h_query_points, struct Point *tree, int qp_n, int tree_n, int k, int *result);

#endif

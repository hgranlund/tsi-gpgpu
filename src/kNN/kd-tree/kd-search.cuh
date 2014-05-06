#ifndef _KD_SEARCH_
#define _KD_SEARCH_
#include "point.h"

#define THREADS_PER_BLOCK_SEARCH 64U
#define MAX_BLOCK_DIM_SIZE 65535U
#define MAX_SHARED_MEM 49152U


__device__ __host__ void cuInitStack(struct SPoint **stack);
__device__ __host__ bool cuIsEmpty(struct SPoint *stack);
__device__ __host__ void cuPush(struct SPoint **stack, struct SPoint value);
__device__ __host__ struct SPoint cuPop(struct SPoint **stack);
__device__ __host__ struct SPoint cuPeek(struct SPoint *stack);
__device__ __host__ void cuInitKStack(KPoint **k_stack, int n);
__device__ __host__ void cuInsert(struct KPoint *k_stack, struct KPoint k_point, int n);
__device__ __host__ struct KPoint cuLook(struct KPoint *k_stack, int n);

__device__ __host__ void cuUpDim(int *dim);

__device__ __host__ void cuKNN(struct Point qp, struct Node *tree, int n, int k, int *result,
                               struct SPoint *stack_ptr, struct KPoint *k_stack_ptr);

void queryAll(struct Point *h_query_points, struct Node *tree, int qp_n, int tree_n, int k, int *result);

#endif

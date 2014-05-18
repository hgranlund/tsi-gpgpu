#ifndef _CU_KD_SEARCH_
#define _CU_KD_SEARCH_
#include <point.h>
#include <stack.h>

#define THREADS_PER_BLOCK_SEARCH 64U
#define MAX_BLOCK_DIM_SIZE 8192U
#define MAX_SHARED_MEM 49152U
#define MIN_NUM_QUERY_POINTS 100U

__device__ __host__ void cuInitStack(struct SPoint **stack);
__device__ __host__ bool cuIsEmpty(struct SPoint *stack);
__device__ __host__ void cuPush(struct SPoint **stack, struct SPoint value);
__device__ __host__ struct SPoint cuPop(struct SPoint **stack);
__device__ __host__ struct SPoint cuPeek(struct SPoint *stack);
__device__ __host__ void cuInitKStack(KPoint **k_stack, int n);
__device__ __host__ void cuInsert(struct KPoint *k_stack, struct KPoint k_point, int n);
__device__ __host__ struct KPoint cuLook(struct KPoint *k_stack);

__device__ __host__ int fastIntegerLog2(int x);

__device__ __host__ float cuDist(struct Point qp, struct Node point);
__device__ __host__ void cuUpDim(int &dim);

__device__ __host__ void cuKNN(struct Point qp, struct Node *tree, int n, int k,
                               struct SPoint *stack_ptr, struct KPoint *k_stack_ptr);

size_t getFreeBytesOnGpu();
int getQueriesInStep(int n_qp, int k, int n);
void getThreadAndBlockCountForQueryAll(int n, int &blocks, int &threads);
size_t getNeededBytesInSearch(int n_qp, int k, int n, int thread_num, int block_num);

void cuQueryAll(struct Point *h_query_points, struct Node *tree, int qp_n, int tree_n, int k, int *result);

#endif

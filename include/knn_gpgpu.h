#ifndef _KNN_GPGPU_
#define _KNN_GPGPU_

#include "point.h"

void buildKdTree(struct Point *points, int n_tree, struct Node *tree);

void queryAll(struct Point *h_query_points, struct Node *h_tree, int n_qp, int n_tree, int k, int *h_result);
void cuQueryAll(struct Point *query_points, struct Node *tree, int n_qp, int n_tree, int k, int *result);
void mpQueryAll(struct Point *query_points, struct Node *tree, int n_qp, int n_tree, int k, int *result);

void knn_brute_force_garcia(float *ref_host, int ref_width, float *query_host, int query_width, int height, int k, float *dist_host, int *ind_host);
void knn_brute_force(float *ref_host, int ref_nb, float *query_host, int dim, int k, float *dist_host, int *ind_host);


// #### Utils
size_t getFreeBytesOnGpu();
void cuSetDevice(int devive);

// Tree build
size_t getNeededBytesForBuildingKdTree(int n_tree);
size_t getTreeSize(int n_tree);

// Search
size_t getNeededBytesForQueryAll(int n_qp, int k, int n_tree);
size_t getNeededBytesInSearch(int n_qp, int k, int n_tree, int thread_num, int block_num);
size_t getSStackSizeInBytes(int n_tree, int thread_num, int block_num);
int getSStackSize(int n_tree);



#endif //  _KNN_GPGPU_

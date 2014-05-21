#ifndef _KNN_GPGPU_
#define _KNN_GPGPU_

#include "point.h"

void buildKdTree(struct Point *points, int n, struct Node *tree);

void cuQueryAll(struct Point *query_points, struct Node *tree, int n_qp, int n_tree, int k, int *result);
void mpQueryAll(struct Point *query_points, struct Node *tree, int n_qp, int n_tree, int k, int *result);

void knn_brute_force_garcia(float *ref_host, int ref_width, float *query_host, int query_width, int height, int k, float *dist_host, int *ind_host);
void knn_brute_force(float *ref_host, int ref_nb, float *query_host, int dim, int k, float *dist_host, int *ind_host);

#endif //  _KNN_GPGPU_

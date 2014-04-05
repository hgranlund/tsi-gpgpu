#ifndef _KERNELS_H_
#define _KERNELS_H_
#include "point.h"


void knn_brute_force_garcia(float *ref_host, int ref_width, float *query_host, int query_width, int height, int k, float *dist_host, int *ind_host);
void knn_brute_force(float *ref_host, int ref_nb, float *query_host, int dim, int k, float *dist_host, int *ind_host);

void build_kd_tree(struct PointS *points, int n, struct Point *tree);
void queryAll(struct Point *query_points, struct Point *tree, int n_qp, int n_tree, int k, int *result);


#endif //  _KERNELS_H_

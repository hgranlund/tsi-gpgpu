KNN GPGPU Documentation
=======================

Includes
--------

#### point.h

Contains definitions of the different point struct data-types used by the knn gpgpu algorithms. the source file is located at /src/kNN.


Members
-------

Examples on usage of the different members can be found in the test files.


#### void build_kd_tree(struct PointS *points, int n, struct Point *tree)

Accepts a list of n PointS. Builds a balanced kd-tree from these points on the GPU, and writes this tree to the Point list tree.


#### void queryAll(struct Point *query_points, struct Point *tree, int n_qp, int n_tree, int k, int *result)

Queries a previously built kd-tree of size n_tree for the k closest neighbors to the points specified in the query_points list of size n_qp. The index location of the k closest points are written to the result array.


#### void knn_brute_force_garcia(float *ref_host, int ref_width, float *query_host, int query_width, int height, int k, float *dist_host, int *ind_host)

Performs a brute force knn-search based on the code written by Garcia.


#### void knn_brute_force(float *ref_host, int ref_nb, float *query_host, int dim, int k, float *dist_host, int *ind_host)

Performs a improved brute force knn-search.
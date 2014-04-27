KNN GPGPU Documentation
=======================

Includes
--------

#### [point.h](https://github.com/hgranlund/tsi-gpgpu/blob/master/src/kNN/point.h)

Contains definitions of the different point struct data-types used by the knn gpgpu algorithms.


Members
-------

#### [void build_kd_tree(struct PointS *points, int n, struct Point *tree)](https://github.com/hgranlund/tsi-gpgpu/blob/master/src/kNN/kd-tree/kd-tree-build.cu)

Accepts a list of n PointS. Builds a balanced kd-tree from these points on the GPU, and writes this tree to the Point list tree.

Example on usage can be found in [this](https://github.com/hgranlund/tsi-gpgpu/blob/master/tests/kNN/kd-tree/time-kd-search.cu) test file.


#### [void queryAll(struct Point *query_points, struct Point *tree, int n_qp, int n_tree, int k, int *result)](https://github.com/hgranlund/tsi-gpgpu/blob/master/src/kNN/kd-tree/kd-search.cu)

Queries a previously built kd-tree of size n_tree for the k closest neighbors to the points specified in the query_points list of size n_qp. The index location of the k closest points are written to the result array.

Example on usage can be found in [this](https://github.com/hgranlund/tsi-gpgpu/blob/master/tests/kNN/kd-tree/time-kd-tree-build.cu) test file.


#### [void knn_brute_force_garcia(float *ref_host, int ref_width, float *query_host, int query_width, int height, int k, float *dist_host, int *ind_host)](https://github.com/hgranlund/tsi-gpgpu/blob/master/src/kNN/brute-force/kNN-brute-force-garcia.cu)

Performs a brute force knn-search based on the code written by Garcia.


#### [void knn_brute_force(float *ref_host, int ref_nb, float *query_host, int dim, int k, float *dist_host, int *ind_host)](https://github.com/hgranlund/tsi-gpgpu/blob/master/src/kNN/brute-force/kNN-brute-force.cu)

Performs a improved brute force knn-search.
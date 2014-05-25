KNN GPGPU Documentation
=======================

Includes
--------

#### [point.h](https://github.com/hgranlund/tsi-gpgpu/blob/master/src/kNN/point.h)

Contains definitions of the different point struct data-types used by the knn gpgpu algorithms.


Members
-------

#### [void buildKdTree(struct Point *points, int n, struct Node *tree)](https://github.com/hgranlund/tsi-gpgpu/blob/master/src/kNN/kd-tree/kd-tree-build.cu)

Accepts a list of n PointS. Builds a balanced kd-tree from these points on the GPU, and writes this tree to the Point list tree.

Example on usage can be found in [this](https://github.com/hgranlund/tsi-gpgpu/blob/master/tests/kNN/kd-tree/time-kd-search.cu) test file.


#### [void cuQueryAll(struct Point *query_points, struct Node *tree, int n_qp, int n_tree, int k, int *result)](https://github.com/hgranlund/tsi-gpgpu/blob/master/src/kNN/kd-tree/kd-search.cu)

Queries a previously built kd-tree of size n_tree for the k closest neighbors to the points specified in the query_points list of size n_qp. The index location of the k closest points are written to the result array.

Example on usage can be found in [this](https://github.com/hgranlund/tsi-gpgpu/blob/master/tests/kNN/kd-tree/time-kd-tree-build.cu) test file.


#### [void mpQueryAll(struct Point *query_points, struct Node *tree, int n_qp, int n_tree, int k, int *result)](https://github.com/hgranlund/tsi-gpgpu/blob/master/src/kNN/kd-tree/kd-search-openmp.cu)

Performes same operations as cuQueryAll, but is parallelized on the CPU using OpenMP instead of CUDA.

Example on usage can be found in [this](https://github.com/hgranlund/tsi-gpgpu/blob/master/tests/kNN/kd-tree/time-kd-search-openmp.cu) test file.


#### [void knn_brute_force_garcia(float *ref_host, int ref_width, float *query_host, int query_width, int height, int k, float *dist_host, int *ind_host)](https://github.com/hgranlund/tsi-gpgpu/blob/master/src/kNN/brute-force/kNN-brute-force-garcia.cu)

Performs a brute force knn-search based on the code written by Garcia.

#### [void knn_brute_force(float *ref_host, int ref_nb, float *query_host, int dim, int k, float *dist_host, int *ind_host)](https://github.com/hgranlund/tsi-gpgpu/blob/master/src/kNN/brute-force/kNN-brute-force.cu)

Performs a improved brute force knn-search.

### Utils

#### [size_t getFreeBytesOnGpu()](https://github.com/hgranlund/tsi-gpgpu/blob/master/src/common/utils.cu)

Return the current amount of free memory on the GPU in bytes.

#### [void cuSetDevice(int device)](https://github.com/hgranlund/tsi-gpgpu/blob/master/src/common/utils.cu)

Sets device as the current device for the calling host thread.


#### [int cuGetDevice()](https://github.com/hgranlund/tsi-gpgpu/blob/master/src/common/utils.cu)

Returns the device on which the active host thread executes the device code.

#### [int cuGetDeviceCount()](https://github.com/hgranlund/tsi-gpgpu/blob/master/src/common/utils.cu)

Returns the number of devices accessible.


#### [size_t getNeededBytesForBuildingKdTree(int n_tree)](https://github.com/hgranlund/tsi-gpgpu/blob/master/src/kNN/kd-tree/kd-tree-build.cu)

Returns needed bytes on GPU to build a tree of size n_tree.

#### [size_t getTreeSize(int n_tree)](https://github.com/hgranlund/tsi-gpgpu/blob/master/src/kNN/kd-tree/kd-tree-build.cu)

Returns the size in bytes of a tree with length n_tree.


#### [size_t getNeededBytesForQueryAll(int n_qp, int k, int n_tree)](https://github.com/hgranlund/tsi-gpgpu/blob/master/src/kNN/kd-tree/cu-kd-search.cu)

Returns needed bytes on GPU to perform a queryAll operation on CUDA.


#### [size_t getNeededBytesInSearch(int n_qp, int k, int n_tree, int thread_num, int block_num)](https://github.com/hgranlund/tsi-gpgpu/blob/master/src/kNN/kd-tree/cu-kd-search.cu)

Returns needed bytes on GPU to perform a queryAll operation on CUDA without taking the tree size into account.




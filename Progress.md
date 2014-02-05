# Progress


## Read
* CUDA by examples.
    * It is a good book to learn the basic of CUDA programming.
* Garcia and E. Debreuve and F. Nielsen and M. Barlaud. k-nearest neighbor search: fast GPU-based implementations and application to high-dimensional feature matching. In Proceedings of the IEEE International Conference on Image Processing (ICIP), Hong Kong, China, September 2010.
    * Garcia's algorithm is based on a brute-force implementation, using Euclidean distance and a modified/parallelized insertion sort algorithm.
* Improving the k-Nearest Neighbour Algorithm with CUDA by Graham Nolan.
    * Improves Garcias algorithm with a bitonic sort
    * Stats that Brute-force is better then kn-tree when implementing kNN on GPU.



## Week 6

### Done
* Implemented bitonic sort on gpu --> Knn-brute-force is reimplemented and  can compute more then ~65000 points.
* Made a test to bitonic sort.


### TODO
* Plot a graph to show knn-brute-force performance.
*



## Week 5

### Done
* Made correctness test for knn_brute_force_garcia.
* Implemented a serial bitonic sort.
* Started to re-implement the brute force algorithm to make it handle more then ~65000 points.  It only need the sorting step to be a working knn-algorithm.

### Todo
* Make the knn brute borce reimplementation complete.
* Optimize the re-implemented knn-algorithm.

## Week 4

### Done

* Made garcias algorithm into a lib.
* Read and understand Garcias algorithm.
* Added gtest.
* Make cmake files.




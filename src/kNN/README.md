The quest for a fast KNN search
===============================

This document is a summary of our most recent (7 February 2014) findings, in the quest for a fast kNN search algorithm. The most up to date information can be found in the [release notes](https://github.com/hgranlund/tsi-gpgpu/tree/master/src/kNN#v13-release-notes) for the most recent version of this project.

Our initial investigation led us to believe that a serial implementation could be as fast as the parallel brute-force solution, for point clouds with fewer than 1 000 000 points, given that both algorithms start with an unordered set of points. Reimplementing the brute-force algorithm with bitonic sort, and optimizing for three dimensions, has shown us that this initial belief was unsupported, and currently the brute force algorithm is faster when starting from a unorganized set of points. When considering repeated querying of the same point cloud, the k-d tree based solution pulls ahead, as most of its running time is spent building the k-d tree for querying. If building the k-d tree could be parallelized this could change. although documented in literature, such an parallelization is still elusive.

In order to make the document more readable, we have included short descriptions of the algorithms used, a short reference to theoretical time complexity. We then go on to list our current results, problematic areas and possible improvements.

The following papers, available in the resources folder, forms the literary basis for our current work.

Related to the brute force approach:
* _Improving the k-Nearest Neighbor Algorithm with CUDA - Graham Nolan_
* _Fast k Nearest Neighbor Search using GPU - Garcia et al._
* _K-nearest neighbor search: fast gpu-based implementations and application to high-dimensional feature matching - Garcia et al._

Related to the k-d tree based approach:
* _Real-Time KD-Tree Construction on Graphics Hardware - Kun Zhou et al._


Brute force based effort
========================

Garcia's base algorithm
-----------------------

Garcia's algorithm is based on a naive brute-force approach. It consists if two steps:

1. Calculate the distance between all reference points and query points.
2. Sort the distances and pick the k smallest distances.

Garcias implementation supports any number of dimensions, reference points and query points (or up to ~65000, number of blocks in the GPU). Due to this feature the algorithm use a lot of extra computation power when only one query point and a small dimensions is selected.

#### Time complexity

Steps:

1. O(n). Every reference point must be evaluated once. Since all calculations are independent, we have a large potential for parallelizing.
2. Insertion sort: O(n^2).


Our reimplementation
--------------------

### Bitonic-sort

Graham Nolan discusses the possibility of improving Garcia's algorithm by reimplementing step two with a bitonic sort. His source code has not been available to us, but he states that the run-time improvements was significant. As well as choosing bitonic sort for the sorting stage of our algorithm, our implementation supports up to 15 000 000 points before memory errors occur, and we have limited the number of dimensions to three.

#### Time complexity

Steps:

1. O(n).
2. Bitonic sort: worst case = O(n*log²(n)), average time ( parallel) = O(log²(n)).


#### Results

Testing the different algorithms for a range of point cloud sizes and a fixed value for k, gave the following results.

![knn-brute-force-vs-serial-k-d-tree](./images/knn-brute-force-vs-serial-k-d-tree.png)

We see that our reimplementation of the brute-force algorithm performs well overall, notably improving on Garcia's implementation (only visible as a short line in the beginning of the graph, due to the restricted number of points it is able to compute). Still more speed is desired before good interactive usage can be achieved.


Test with n =8 388 608:

* Memory transfer:  21.1 ms.
* Calculate all distances: 2.5 ms.
* Bitonic sort:  176 ms.
* Total: 200 ms.

#### Min-Reduce

An other possibility to improve step 2 is to use a reduce operation to get the smallest distances. This can be done k times to get the k smallest values.

#### Time complexity

Steps:

1. O(n).
2. Min-reduce: k* log²(n)).

#### Memory optimalisation

We have done some memory optimization based on a [presentation](https://github.com/hgranlund/tsi-gpgpu/blob/master/resources/kNN/reduction.pdf) from Nvidia.

The optimizations include:

* Shared memory utilization.
* Sequential Addressing.
* Complete for-loop Unrolling.

![Comparison between knn-brute-force-reduce with and without memory optimizations (k=10).](./images/knn-brute-force-reduce-memory-opt.png)

#### Results

Test results of n = 8 388 608 with no memory optimization:

*  Memory transfer:  21.1 ms.
*  Calculate all distances: 2.5 ms
*  One min-reduce step : 4.8 ms.
*  Total time: (23.7 + k*4.8) ms.

Test results of n = 8 388 608 with memory optimization:

*  Memory transfer:  21.1 ms.
*  Calculate all distances: 2.5 ms
*  One min-reduce step : 1.7 ms.
*  Total time: (23.7 + k*1.7) ms.


![Comparison between bitonic and reduce (k=10).](./images/BitonicVSreduce.png)

fig: Brute-force - k = 10


#### Possible improvements

* Memory improvements. Use shared memory and texture memory.
* Modify bitonic sort, so do not need to sort all points. We can split the distance array to fit into the GPU blocks, move the smallest values in each block, then sort the moved values. ~O((n/b)* b*log²(b)) subsetof O(n/b), b = Number of threads in each block, n= number of reference points
* Replace bitonic sort with min reduce. O(k*log²(n)).


k-d tree based effort
=====================

A k-d tree can be thought of as a binary search tree for graphical data. A few different variations exist, but we will focus our explanation around a 2D example, storing point data in all nodes. The plane is split into two sub-planes along one of the axis (in our example the y-axis) and all the nodes are sorted as to whether they belong to the left or right of this split. To determine the left and right child of the root node, the two sub-planes are again split at an arbitrary point, this time cycling to the next axis (in our example the x-axis) and the

![2d-k-d-tree](./images/Kdtree_2d.png)

In order to build a k-d tree for 3D space, you simply cycle through the three dimensions, instead of two.

Given the previous splits and selection of nodes, the resulting binary tree would be as shown in the illustration under. (All illustrations gratuitously borrowed from [Wikipedia](http://en.wikipedia.org/wiki/K-d_tree))

![corresponding-binary-tree](./images/Tree_0001.png)

Given that the resulting binary tree is balanced, we get an average search time for the closest neighbor in O(log² n) time. For values of k << n, the same average search time can be achieved, with minimal changes to the algorithm, when searching for the k closest neighbors. It is known from literature that balancing the tree can be achieved by always splitting on the meridian node. Building a k-d tree in this manner takes O(kn log² n) time.

Interested readers is encouraged to look at the paper "Multidimensional binary search trees used for associative searching" by Jon Louis Bentley, where kd-trees first was described.


The serial base algorithm
-------------------------

1. Build a balanced k-d tree from the point cloud.
2. Query the tree for different sets of neighbors.

#### Time complexity

Steps:

1. O(n log² n). Achieving this speed is dependent on an efficient algorithm for finding the meridian.
2. Approximately O(log² n), but dependent on size of k.

#### Results

![serial-k-d-tree-breakdown](./images/serial-k-d-tree-breakdown.png)

As expected, almost all the time is spent building the tree. Querying for the closest neighbor in the largest tree took less than 0.0015 ms, but 9 seconds is a long time to wait for the tree to build.

The paper _Real-Time KD-Tree Construction on Graphics Hardware - Kun Zhou et al._ offers interesting, although slightly complex, ideas to an efficient parallelization of k-d tree construction. I order to save time, a good amount of time was spent searching for, and trying out, different open source implementations based on this paper. This search was unsuccessful. All the implementations we managed to find was problematic due to lack of updates, often not updated since 2011, and still running on CUDA 4.1, lack of documentation, lack of generalization or dubious source code.

A more uplifting find was several references to _Real-Time KD-Tree Construction on Graphics Hardware_ in material published by Nvidia, regarding their proprietary systems for ray tracing. A graphics rendering technique often reliant on k-d trees, and indeed dependent on high performance.


Our reimplementation
--------------------

Finding a way to represent the kd-tree boils down to representing a binary tree. This can be done in several ways, but we had some criteria for our representation:

* The kd-tree should be memory-efficient, at best explicitly storing only the actual point coordinates. This since we want to store the largest possible amount of points on the smaller memory of the GPU.
* The kd-tree representation should be easy to split, distribute and join, since we want to build it on several independent processes.

After a bit of research and trial and error, we choose to represent the kd-tree as a array of structs, representing points, with an x, y and z coordinate stored in an short array. In order to turn this array into a binary tree, the following scheme was derived. Given an array, the root node of that array is the midmost element (the leftmost of the two, given an even number of elements). To find the left child of the root, you simply find the midmost element of the left sub-array, and likewise for the right child. This representation have some advantages and drawbacks.

Advantages:

* Can represent binary trees where not all leaf-nodes are present. This is the minimal requirement for representing perfectly balanced kd-trees.
* Joining or splitting subtrees is as simple as appending or splitting arrays.
* Minimal memory overhead, kd-trees can even be built in-place on the array.

Drawbacks:

* Cannot represent imperfectly balanced kd-trees. This mens that the median cannot be calculated through heuristic methods, but enforces a tree optimized for fast queries.
* Location of children and parents have to be recalculated for all basic traversing of the tree, with may reduce the performance of queries on the tree. In order to eliminate this drawback, a index cache is computed before the search is performed.

Given this representation of the kd-tree, the base kd-tree building algorithm can be expanded to the following:

Steps:

1. Find the exact median (of the current split dimension) in this sub-array, and swap it to the midmost place.
2. Go through all elements in the list, and swap elements, such that all elements lower than the median is placed to the left, and all elements higher than the median is placed to the right.
3. Repeat for the new sub-arrays, until the entire tree is built.

#### Results

_The serial kd-tree build times was similar or better than the base algorithm, so it is not discussed here_

![awg-query-time-old-vs-new](./images/awg-query-time-old-vs-new.png)

The graph shows that our serial implementation of search in this data structure, given a pre-calculated index cache, is as fast, or slightly faster than the base algorithm. The possibility of improving the search even more, by storing all calculated distances in a distance cache was also explored, but we can see from the graph that the overhead associated with this operation did outweigh the benefits.

Some instability is apparent in the graph, but this is probably due to the author running other programs in the background when performing the test.

![n-query-time-old-vs-new](./images/n-query-time-old-vs-new.png)

The trend is exaggerated when considering N searches. Searching with an index cache is overall fastest, and gives a total search time of ~17 s for 14 million searches in a point cloud of 14 million.


#### Possible improvements

* Run several queries in parallel. Easy to implement, as parallelization is trivial due to independent queries, but is dependent on efficient transfer and storage of the kd-tree on the GPU.
* Explore performance on a variable number of k.


Parallel improvements
---------------------

As we noted in the previous section, the kd-tree build process is by far the most expensive operation, and we would save a lot of time by managing to parallelize this operation. In order to do this, we have to look a bit closer at the different steps of the kd-tree build algorithm.

Steps:

1. Find the median of the points along a specified axis. This median point becomes the value of the current node.
2. Sort all points with lower values than the median to the left of the median, and all the points with higher values than the median to the right.
3. Perform this algorithm recursively on the left and right set of nodes.

Several strategies can be used to parallelize this code. We can perform the recursive calls as a increasing number of different independent processes. We can also use a parallel algorithm for finding the median in each recursive call. Both strategies can be used in conjunction with each other. The parallel algorithm for finding the median can be used to speed up the early iterations, where we do not have the possibility of calculating several sub-trees in parallel, as well as speeding up the calculation on lather calculations, by utilizing the large number of concurrent threads available in each parallel process.

Different parallel algorithms for finding the median was considered. First we tried to reuse the implementation of bitonic sort. Given a sorted list you can find the median directly, by simply looking at the midmost element of the array. Unfortunately this strategy proved unsuccessful, as re-purposing the bitonic algorithm for such an task proved difficult. We also have the inherent downside of sorting a list in order to find the median, since O(n) algorithms for finding the median exist, compared to the O(n log(n)) time required by sorting.

The existing O(n) algorithms for finding the median is mostly based on a more generic problem, namely selection or [kth order statistic algorithms](http://en.wikipedia.org/wiki/Selection_algorithm). Quick select and radix select to two of the best known selection algorithms in serial. They have both an average time complexity of O(n), witch makes them a good candidates. The difference between then is the constant time penalty. The radix sort have a more exact time complexity of O(bn), where b is the number of bits in each number. While the penalty for quick select is based on n, and if bad pivot elements are chosen the worst case performance is O(n²). We choose to start implementing radix sort based on the results from [*Radix Selection Algorithm for the kth Order Statistic*](https://github.com/hgranlund/tsi-gpgpu/blob/master/resources/kNN/radix_select.pdf). Based on the constant time penalties radix sort would also be the best candidate for large number of elements (n).

The radix select is based on a bitwise partitioning, much like radix sort. In each step, elements are partitioned in two subgroups based on the current bit. Then the subgroup that contains the median is determined, and the search continue in that subgroup until the median is found.

![An illustration of radix selectionn](./images/Radix_select.png)


It is hard to use all the parallel power of cuda in this algorithm. The reason is that the problem is divided in three different types; partition one huge list, partition some middle sized list and partition many small lists. This is the reason why we have chosen to use three different implementation of k'th order statistic. The constant time penalties of the two algorithms we have chosen give us a clear indication the radix select is best on large lists while quick select is best on small lists.

#### Results

![gpu-vs-cpu-build-time](./images/gpu-vs-cpu-build-time.png)

We see that the parallel implementation performs better than the base serial implementation, building a tree of 14 million points in just over 5 seconds, compared to just under 10 required by the serial algorithm. Still we regard this as a quite rough implementation, in need of more tuning to really bring out the speed potential. The potential for parallelizing the workload for the first and last iterations have not been fully developed. This is due to the implementation forcing one version of the radix select algorithm being to work on all problem sizes. This is not optimal for dividing cuda resources, and as a result, we get high penalties when the problem size reaches "unsuitable" values.

We also see a couple of large "jumps" in the graph. This happens when the number of elements passes a power of two and the height of the resulting kd-tree increase. The height increase hits the implementation at its weakest.

Tuning the algorithm to alternate between radix select and quick select, eliminates this problem, as is visible in the graph for GPU v1.1. This removes the penalty for calculating the median at "unsuitable" problem sizes, giving an build time of ~2.4 seconds for 14 million points, compared to the ~9 seconds required by the serial implementation, or the ~5.2 seconds required by the old parallel implementation.


#### Memory usage

To analyses the space complexity of the kd-tree build and search algorithm, we have made an theoretical calculation of both algorithms GPU memory consumption, and tested it against results from a GeForce 560ti and a Nvidia grid K520 (amazon web service delved).

It is important to note that the only hard memory limitation is related to building the tree, as a search for N query-points can be performed in several operations. If you e.g. run into memory limitations when searching for 10^8 query-points, you can simply perform two searches on 5^8 query-points to get around the limitation. Loading the pre-built kd-tree on the GPU for searching, and performing one query for a low value of k, will always consume less memory than building the actual kd-tree.

**Kd-tree-build**

The memory consumption for the kd-tree build is only depended on the number of points (n) and the theoretical consumption rate grows linearly as O(36n) ⊂ O(n).

![Memory usage of kd-tree-build](./images/memory-usage-build.png)

We see that the estimation fit the real consumption almost perfectly, and with this memory model, we can easily estimate the GPU memory requirements for different problem sizes.

Given that a customer wants to perform a knn-search on a point cloud of 100 million, he or she would need a GPU with at least 3.6 Gb of spare memory. Under we have tabulated what maximum problem sizes you would expect to be able to run on a selection of Nvidia graphics cards:

| Nvidia GPU   | Available memory | Maximum problem size |
| ------------ |:----------------:| --------------------:|
| GTX TITAN    | 6144 MB          | 1.79E+08             |
| GTX 780      | 3072 MB          | 8.95E+07             |
| GTX 770      | 2048 MB          | 5.97E+07             |
| Quadro K6000 | 12288 MB         | 3.58E+08             |
| Quadro K5000 | 4096 MB          | 1.19E+08             |
| Quadro K4000 | 3072 MB          | 8.95E+07             |
| Tesla K40    | 12288 MB         | 3.58E+08             |
| Tesla K20    | 5120 MB          | 1.49E+08             |

These numbers should be read as rough estimates, as each card is expected to have internal processes requiring an unspecified constant amount of the available memory, therefor lovering the maximum problem size possible to run on these cards in practice. It is also worth to mention that when buying a GPU for GPGPU tasks, other performance characteristics is equally, or more, important. 

**Kd-search**

The kd-search is used to query every point against each other. It has a theoretical memory consumption rate at O((40+4k)n) ⊂ O(kn). The consumption is therefore depended on the number of points (n) and the number of neighbors (k).

![Memory usage of kd-search](./images/memory-usage-kd-search.png)

Also in this case our estimation fit the real consumption with a high degree of accuracy.


#### Further work

* Look at memory optimization.
* Improve utiliti methods like: accumulateindex, minReduce.
* Forloop Unrolling.


V1.3 Release notes
------------------

Version 1.3 introduces a couple of new features. Firstly the tree-building algorithm has been updated to also cache the location of the different children of each node in the tree. A small bug related to partition of the point-cloud list was also ironed out. This gives a tree building algorithm whit the following runtime results, comparable with the previous versions of this implementation:

![construction_v13_gtx_560](./images/construction_v13_gtx_560.png)

After a surprising amount of fiddling, querying for a large number of query-points was finally parallelized in version 1.3. This gave improved performance when querying many times in the same point cloud. 14 million queries in a point cloud of 14 million points can now be done on average in ~8 seconds. Compared to the estimated ~17 seconds needed by the serial implementation.

![n_queries_v13_gtx_560](./images/n_queries_v13_gtx_560.png)

Unfortunately, this early parallelization gave us quite unstable results. The data in the graph is based on ten timing runs, where the average, minimum and maximum values are collected in three series. As we can see, there is quite a big spread between the best and worst results. Looking at the raw data points for the ten series confirms that this is not a problem caused by a small number of outliers:

![scatter_n_queries_v13_gtx_560](./images/scatter_n_queries_v13_gtx_560.png)

Some instability would be expected, as the amount of pruning that can be achieved when searching for points in the kd-tree is dependent on the value you search for, but this amount of spread was unexpected. This behavior should be investigated further, as it seems to be indicating some kind of implementation error.

Combining the results from the search and the tree-building, gives the following runtime for a sequence of building and N queries:

![constructionand_n_queries_v13_gtx_560](./images/constructionand_n_queries_v13_gtx_560.png)


Work-plan for next week
-----------------------

Major improvements of the tree-building implementation is not to bee expected, so further work will focus on improving existing code through refactoring. The unstable behavior of the parallel search should be studied in more detail, in conjunction with expanding it to work with larger values of k, and improving the memory usage of this algorithm. Finally it might be beneficial to spend some time on getting the project to build at TSI HQ, so version 1.3 can be studied in more detail by Alok.

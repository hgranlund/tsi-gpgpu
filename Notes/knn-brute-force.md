# Knn brute-force

## Garcia

Garcia's algorithm is based on a naive brute-force approach. It consists if two steps:

1. Calculate the distance between all reference points and query points.
2. Sort the distances and pick the k smallest distances.

Garcias algorithm supports any number of dimensions, reference points and query points (or up to ~65000, number of blocks in the GPU). Due to this feature the algorithm use a lot of extra computation power when only one query point and a small dimensions is selected.

### Time complexity

Steps:

1. O(n). Every reference point must be evaluated once. Huge positional for parallelizing.
2. Insertion sort: O(n^2).


## Graham Nolan

He is talking about improving Garcia's algorithm by reimplementing step two with a bitonic sort.

We have not been able to get the source code, but he said the improvement was significant.


### Time complexity

Steps:

1. O(n).
2. Bitonic sort: worst case = O(n*log²(n)), average time ( parallel) = O(log²(n)).


## Our reimplementation.

This algorithm uses the same steps as Garcia and Nolan, but they have been reimplemented to support more points. (Got memory error over 15 000 000 points).



### Time complexity

Steps:

1. O(n).
2. Bitonic sort: worst case = O(n*log²(n)), average time ( parallel) = O(log²(n)).


### Possible improvements

* Memory improvements. Use shared memory and texture memory.
* Modify bitonic sort, so do not need to sort all the points. We can split the distance array to fit into the GPU blocks, move the smallest values in each block, then sort the moved values. ~O((n/b)* b*log²(b)) subsetof O(n/b), b = Number of threads in each block(constant), n= number of reference points
* Replace bitonic sort with min reduce. O(k*log²(n)).


#ifndef _KD_TREE_NAIVE_
#define _KD_TREE_NAIVE_
#include "point.h"

void build_kd_tree(Point *points, int n);
int midpoint(int lower, int upper);
void swap(Point *points, int a, int b);
float quick_select(int k, Point *points, int lower, int upper, int dim);
void center_median(Point *points, int lower, int upper, int dim);
int h_index(int i, int j, int n);

__device__
int index(int i, int j, int n);
#endif

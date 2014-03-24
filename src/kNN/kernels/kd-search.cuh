#ifndef _KD_SEARCH_
#define _KD_SEARCH_
#include "point.h"

int store_locations(Point *tree, int lower, int upper, int n);
int nn(float *qp, Point *tree, int dim, int index);
int mid(int lower, int upper);

#endif
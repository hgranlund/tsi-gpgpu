#ifndef _TEST_COMMON_
#define _TEST_COMMON_

#include <point.h>
#include <stdio.h>
#include <cuda.h>
#include <helper_cuda.h>

#define checkCudaErrors(val)           check ( (val), #val, __FILE__, __LINE__ )

void populatePoints(struct Point *points, int n);
void populatePointSs(struct PointS *points, int n);
void populatePointSRosetta(struct PointS *points, int n);

#endif

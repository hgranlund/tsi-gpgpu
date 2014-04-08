#ifndef _TEST_COMMON_
#define _TEST_COMMON_
#include <point.h>
#include <stdio.h>
#include <helper_cuda.h>
#include "gtest/gtest.h"

void populatePoints(Point *points, int n);

void populatePointSs(PointS *points, int n);

void cudaStartTiming(cudaEvent_t &start, cudaEvent_t &stop, float &elapsed_time);

void cudaStopTiming(cudaEvent_t &start, cudaEvent_t &stop, float &elapsed_time);

void printCudaTiming(float elapsed_time, float bytes, int n);

void ASSERT_TREE_EQ(Point *expected_tree, Point *actual_tree, int n);


#endif

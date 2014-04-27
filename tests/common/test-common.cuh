#ifndef _TEST_COMMON_
#define _TEST_COMMON_

#include <point.h>
#include <stdio.h>
#include <cuda.h>
#include <helper_cuda.h>
#include "gtest/gtest.h"

#define checkCudaErrors(val)           check ( (val), #val, __FILE__, __LINE__ )

void populatePoints(struct Point *points, int n);
void populatePointSs(struct PointS *points, int n);
void populatePointSRosetta(struct PointS *points, int n);

void cudaStartTiming(cudaEvent_t &start, cudaEvent_t &stop, float &elapsed_time);
void cudaStopTiming(cudaEvent_t &start, cudaEvent_t &stop, float &elapsed_time);
void printCudaTiming(float elapsed_time, float bytes, int n);

double WallTime ();
void printTree(struct Point *tree, int level, int root);

void readPoints(const char *file_path, int n, struct PointS *points);

void ASSERT_TREE_EQ(struct Point *expected_tree, struct Point *actual_tree, int n);
void ASSERT_TREE_LEVEL_OK(PointS *points, int *steps, int n, int p, int dim);
void ASSERT_KD_TREE(struct Point *tree, int n);
#endif

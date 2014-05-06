#ifndef _TEST_COMMON_
#define _TEST_COMMON_

#include <point.h>
#include <stdio.h>
#include "gtest/gtest.h"


void populatePoints(struct Node *points, int n);
void populatePointSs(struct Point *points, int n);
void populatePointSRosetta(struct Point *points, int n);

double WallTime ();
void printTree(struct Node *tree, int level, int root);

void n_step(int *steps, int n, int h);

void readPoints(const char *file_path, int n, struct Point *points);

void ASSERT_TREE_EQ(struct Node *expected_tree, struct Node *actual_tree, int n);
void ASSERT_TREE_LEVEL_OK(struct Point *points, int *steps, int n, int p, int dim);
void ASSERT_KD_TREE(struct Node *tree, int n);
#endif

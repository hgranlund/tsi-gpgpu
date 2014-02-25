#include "kd-tree-naive.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <helper_cuda.h>
#define checkCudaErrors(val)           check ( (val), #val, __FILE__, __LINE__ )

void print_tree1(float *tree, int level, int lower, int upper, int n)
{
    if (lower >= upper)
    {
        return;
    }

    int i, r = midpoint(lower, upper);

    printf("|");
    for (i = 0; i < level; ++i)
    {
        printf("--");
    }
    printf("(%3.1f, %3.1f, %3.1f)\n", tree[index(r, 0, n)], tree[index(r, 1, n)], tree[index(r, 2, n)]);

    print_tree1(tree, 1 + level, lower, r, n);
    print_tree1(tree, 1 + level, r + 1, upper, n);
}

int index(int i, int j, int n)
{
    return i + j * n;
}

void swap(float *points, int a, int b, int n)
{
    float tx = points[index(a, 0, n)],
    ty = points[index(a, 1, n)],
    tz = points[index(a, 2, n)];
    points[index(a, 0, n)] = points[index(b, 0, n)], points[index(b, 0, n)] = tx;
    points[index(a, 1, n)] = points[index(b, 1, n)], points[index(b, 1, n)] = ty;
    points[index(a, 2, n)] = points[index(b, 2, n)], points[index(b, 2, n)] = tz;
}

int midpoint(int lower, int upper)
{
    return (int) floor((upper - lower) / 2) + lower;
}

float quick_select(int k, float *points, int lower, int upper, int dim, int n)
{
    int pos, i,
    left = lower,
    right = upper - 1;

    float pivot;

    while (left < right)
    {
        pivot = points[index(k, dim, n)];
        swap(points, k, right, n);
        for (i = pos = left; i < right; i++)
        {
            if (points[index(i, dim, n)] < pivot)
            {
                swap(points, i, pos, n);
                pos++;
            }
        }
        swap(points, right, pos, n);
        if (pos == k) break;
        if (pos < k) left = pos + 1;
        else right = pos - 1;
    }
    return points[index(k, dim, n)];
}

void center_median(float *points, int lower, int upper, int dim, int n)
{
    int i, r = midpoint(lower, upper);

    float median = quick_select(r, points, lower, upper, dim, n);

    for (i = lower; i < upper; ++i)
    {
        if (points[index(i, dim, n)] == median)
        {
            swap(points, i, r, n);
            return;
        }
    }
}

void balance_branch(float *points, int lower, int upper, int dim, int n)
{
    printf("n=%d, lower = %d, upper =%d\n", n, lower, upper);
    if (lower >= upper) return;

    int i, r = midpoint(lower, upper);

    center_median(points, lower, upper, dim, n);

    upper--;

    for (i = lower; i < r; ++i)
    {
        if (points[index(i, dim, n)] > points[index(r, dim, n)])
        {
            while (points[index(upper, dim, n)] > points[index(r, dim, n)])
            {
                upper--;
            }
            swap(points, i, upper, n);
        }
    }
}

void build_kd_tree(float *points, int n)
{
    int i, j, p, step,
    h = ceil(log2((float)n + 1) - 1);
    for (i = 0; i < h; ++i)
    {
        p = pow(2, i);
        step = (int) floor(n / p);

        for (j = 0; j < p; ++j)
        {
            balance_branch(points, (1 + step) * j, step + (1 + j) + j, i%3, n);
        }
        printf("p = %d,  i=%d \n", p,i);
        print_tree1(points, 0, 0, n, n);
        printf("==================\n");

    }
    return;
}

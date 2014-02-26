#include "kd-tree-naive.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <helper_cuda.h>
#define checkCudaErrors(val)           check ( (val), #val, __FILE__, __LINE__ )



__constant__  size_t* pitch;



__device__ __host__
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

void build_kd_tree(float *h_points, int n)
{
    float* d_points;
    int i, j, step, h, dim = 3;
    size_t d_pitch, h_pitch = n*sizeof(float);

    h = ceil(log2((float)n + 1) - 1);

    checkCudaErrors(
        cudaMallocPitch(&d_points, &d_pitch, sizeof(float)*n, dim));
    checkCudaErrors(cudaMemcpyToSymbol(pitch, &d_pitch, sizeof(size_t)));


    checkCudaErrors(
            cudaMemcpy2D(d_points, d_pitch, h_points, h_pitch, n*sizeof(float), dim, cudaMemcpyHostToDevice));




    checkCudaErrors(
            cudaMemcpy2D(h_points, h_pitch, d_points, d_pitch, n*sizeof(float), dim, cudaMemcpyDeviceToHost));


    for (i = 0; i < h; ++i)
    {
        step = (int) floor(n / pow(2, i)) + 1;
        for (j = 0; j < n; j+=step)
        {
            balance_branch(h_points, j, j+step-1, i%dim, n);
        }
    }

    checkCudaErrors(cudaFree(d_points));
}

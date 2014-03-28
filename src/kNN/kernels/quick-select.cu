#include "quick-select.cuh"
#include <stdio.h>

__device__ void cuPointSwap(Point *p, int a, int b)
{
    Point temp = p[a];
    p[a] = p[b], p[b] = temp;
}

template <int maxStep> __global__
void cuQuickSelectShared(Point *points, int step, int p, int dir)
{
    __shared__ Point ss_points[maxStep * THREADS_PER_BLOCK_QUICK];
    Point *s_points = ss_points;
    float pivot;
    int pos, i, left, right,
        n = step,
        list_in_block = p / gridDim.x,
        tid = threadIdx.x,
        m;

    points += list_in_block * blockIdx.x * (1 + step);
    points += (1 + step) * tid;
    s_points += (tid * maxStep);

    while ( tid < list_in_block)
    {
        n = step - tid;
        m = n >> 1;   // same as n/2;
        for (i = 0; i < n; ++i)
        {
            s_points[i] = points[i];
        }
        left = 0;
        right = n - 1;
        while (left < right)
        {
            pivot = s_points[m].p[dir];
            cuPointSwap(s_points, m, right);
            for (i = pos = left; i < right; i++)
            {
                if (s_points[i].p[dir] < pivot)
                {
                    cuPointSwap(s_points, i, pos);
                    pos++;
                }
            }
            cuPointSwap(s_points, right, pos);
            if (pos == m) break;
            if (pos < m) left = pos + 1;
            else right = pos - 1;
        }
        for (i = 0; i < n; ++i)
        {
            points[i] = s_points[i];
        }
        tid += blockDim.x;
        points += blockDim.x + blockDim.x * step;
    }
}

__global__
void cuQuickSelectGlobal(Point *points, int step, int p, int dir)
{
    int pos, i,
        list_in_block = p / gridDim.x,
        tid = threadIdx.x,
        left,
        right,
        n = step,
        m;
    points += list_in_block * blockIdx.x * n;
    points += (1 + step) * tid;
    float pivot;
    while ( tid < list_in_block)
    {
        n = step - tid;
        m = n >> 1;   // same as n/2;
        left = 0;
        right = n - 1;
        while (left < right)
        {
            pivot = points[m].p[dir];
            cuPointSwap(points, m, right);
            for (i = pos = left; i < right; i++)
            {
                if (points[i].p[dir] < pivot)
                {
                    cuPointSwap(points, i, pos);
                    pos++;
                }
            }
            cuPointSwap(points, right, pos);
            if (pos == m) break;
            if (pos < m) left = pos + 1;
            else right = pos - 1;
        }
        tid += blockDim.x;
        points += blockDim.x + blockDim.x * step;
    }
}

void quickSelectAndPartition(Point *points, int n , int p, int dir)
{
    int numBlocks, numThreads;
    getThreadAndBlockCountForQuickSelect(n, p, numBlocks, numThreads);
    if (n > 16)
    {
        cuQuickSelectGlobal <<< numBlocks, numThreads>>>(points, n, p, dir);
    }
    else if (n > 8)
    {
        if (16 * sizeof(Point) * numThreads < MAX_SHARED_MEM)
        {
            quickSelectShared(points, n, p, dir, 16, numBlocks, numThreads);
        }
        else
        {
            cuQuickSelectGlobal <<< numBlocks, numThreads>>>(points, n, p, dir);
        }
    }
    else if (n > 4)
    {
        if (8 * sizeof(Point) * numThreads < MAX_SHARED_MEM)
        {
            quickSelectShared(points, n, p, dir, 8, numBlocks, numThreads);
        }
        else
        {
            cuQuickSelectGlobal <<< numBlocks, numThreads>>>(points, n, p, dir);
        }
    }
    else
    {
        if (4 * sizeof(Point) * numThreads < MAX_SHARED_MEM)
        {
            quickSelectShared(points, n, p, dir, 4, numBlocks, numThreads);
        }
        else
        {
            cuQuickSelectGlobal <<< numBlocks, numThreads>>>(points, n, p, dir);
        }
    }
}



void quickSelectShared(Point *points, int n, int p, int dir, int size, int numBlocks, int numThreads)
{
    if (size > 16)
    {
        cuQuickSelectGlobal <<< numBlocks, numThreads>>>(points, n, p, dir);
    }
    else if (size > 8)
    {
        cuQuickSelectGlobal <<< numBlocks, numThreads>>>(points, n, p, dir);
        // cuQuickSelectShared<16> <<< numBlocks, numThreads>>>(points, n, p, dir);
    }
    else if (size > 4)
    {
        cuQuickSelectGlobal <<< numBlocks, numThreads>>>(points, n, p, dir);
        // cuQuickSelectShared<8> <<< numBlocks, numThreads>>>(points, n, p, dir);
    }
    else
    {
        cuQuickSelectShared<4> <<< numBlocks, numThreads>>>(points, n, p, dir);
    }
}


void getThreadAndBlockCountForQuickSelect(int n, int p, int &blocks, int &threads)
{
    threads = min(THREADS_PER_BLOCK_QUICK, p);
    blocks = p / threads;
    blocks = min(MAX_BLOCK_DIM_SIZE, blocks);
    blocks = max(1, blocks);
}

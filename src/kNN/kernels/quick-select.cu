#include <quick-select.cuh>
#include <stdio.h>


#define THREADS_PER_BLOCK 1024U
#define MAX_BLOCK_DIM_SIZE 65535U


__device__ void cuPointSwap(Point *p, int a, int b)
{
    Point temp = p[a];
    p[a] = p[b], p[b] = temp;
}

template <int maxStep> __global__
void cuQuickSelectShared(Point *points, int n, int p, int dir)
{
    __shared__ Point ss_points[maxStep * 128];
    Point *s_points = ss_points;
    float pivot;
    int pos, i, left, right,
        listInBlock = p / gridDim.x,
        tid = threadIdx.x,
        m = n >> 1;   // same as n/2;

    points += listInBlock * blockIdx.x * n;
    points += n * tid;
    s_points += (tid * maxStep);
    while ( tid < listInBlock)
    {
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
        points += n * blockDim.x;
    }
}

__global__
void cuQuickSelectGlobal(Point *points, int n, int p, int dir)
{
    int pos, i,
        listInBlock = p / gridDim.x,
        tid = threadIdx.x,
        m = n >> 1;   // same as n/2;
    points += listInBlock * blockIdx.x * n;
    points += n * tid;
    float pivot;
    while ( tid < listInBlock)
    {
        int
        left = 0,
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
        points += n * blockDim.x;
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
        cuQuickSelectShared<16> <<< numBlocks, numThreads>>>(points, n, p, dir);
    }
    else if (size > 4)
    {
        cuQuickSelectShared<8> <<< numBlocks, numThreads>>>(points, n, p, dir);
    }
    else
    {
        cuQuickSelectShared<4> <<< numBlocks, numThreads>>>(points, n, p, dir);
    }
}


void getThreadAndBlockCountForQuickSelect(int n, int p, int &blocks, int &threads)
{
    threads = 128;
    blocks = p / threads;
    blocks = min(MAX_BLOCK_DIM_SIZE, blocks);
    blocks = max(1, blocks);
}

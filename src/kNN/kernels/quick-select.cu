#include "quick-select.cuh"
#include <stdio.h>

__device__ void cuPointSwap(Point *p, int a, int b)
{
    Point temp = p[a];
    p[a] = p[b], p[b] = temp;
}

template <int maxStep> __global__
void cuQuickSelectShared(Point *points, int *steps, int p, int dir)
{
    __shared__ Point ss_points[maxStep * THREADS_PER_BLOCK_QUICK];
    Point *s_points = ss_points, *l_points;
    float pivot;
    int pos, i, left, right, step_num,
        n,
        list_in_block = p / gridDim.x,
        tid = threadIdx.x,
        m;

    s_points += (tid * maxStep);

    while ( tid < list_in_block)
    {
        step_num = tid * blockIdx.x + tid;
        l_points = points + steps[step_num * 2];
        n = steps[step_num * 2 + 1] - steps[step_num * 2];
        m = n >> 1;   // same as n/2;
        for (i = 0; i < n; ++i)
        {
            s_points[i] = l_points[i];
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
            l_points[i] = s_points[i];
        }
        tid += blockDim.x;
    }
}

__global__
void cuQuickSelectGlobal(Point *points, int *steps, int p, int dir)
{
    int pos, i,
        list_in_block = p / gridDim.x,
        tid = threadIdx.x,
        left,
        right,
        step_num,
        n,
        m;

    Point  *l_points;
    float pivot;
    while ( tid < list_in_block)
    {
        step_num = tid * blockIdx.x + tid;
        l_points = points + steps[step_num * 2];
        n = steps[step_num * 2 + 1] - steps[step_num * 2];
        m = n >> 1;   // same as n/2;
        left = 0;
        right = n - 1;
        while (left < right)
        {
            pivot = l_points[m].p[dir];
            cuPointSwap(l_points, m, right);
            for (i = pos = left; i < right; i++)
            {
                if (l_points[i].p[dir] < pivot)
                {
                    cuPointSwap(l_points, i, pos);
                    pos++;
                }
            }
            cuPointSwap(l_points, right, pos);
            if (pos == m) break;
            if (pos < m) left = pos + 1;
            else right = pos - 1;
        }
        tid += blockDim.x;
    }
}

void quickSelectAndPartition(Point *d_points, int *d_steps, int n , int p, int dir)
{
    int numBlocks, numThreads;
    n = n / p;
    getThreadAndBlockCountForQuickSelect(n, p, numBlocks, numThreads);
    if (n > 16)
    {
        cuQuickSelectGlobal <<< numBlocks, numThreads>>>(d_points, d_steps, p, dir);
    }
    else if (n > 8)
    {
        if (16 * sizeof(Point) * numThreads < MAX_SHARED_MEM)
        {
            quickSelectShared(d_points, d_steps, p, dir, 16, numBlocks, numThreads);
        }
        else
        {
            cuQuickSelectGlobal <<< numBlocks, numThreads>>>(d_points, d_steps, p, dir);
        }
    }
    else if (n > 4)
    {
        if (8 * sizeof(Point) * numThreads < MAX_SHARED_MEM)
        {
            quickSelectShared(d_points, d_steps, p, dir, 8, numBlocks, numThreads);
        }
        else
        {
            cuQuickSelectGlobal <<< numBlocks, numThreads>>>(d_points, d_steps, p, dir);
        }
    }
    else
    {
        if (4 * sizeof(Point) * numThreads < MAX_SHARED_MEM)
        {
            quickSelectShared(d_points, d_steps, p, dir, 4, numBlocks, numThreads);
        }
        else
        {
            cuQuickSelectGlobal <<< numBlocks, numThreads>>>(d_points, d_steps, p, dir);
        }
    }
}



void quickSelectShared(Point *d_points, int *d_steps, int p, int dir, int size, int numBlocks, int numThreads)
{
    if (size > 16)
    {
        // cuQuickSelectGlobal <<< numBlocks, numThreads>>>(d_points, d_steps, p, dir);
        cuQuickSelectShared<32> <<< numBlocks, numThreads>>>(d_points, d_steps, p, dir);
    }
    else if (size > 8)
    {
        // cuQuickSelectGlobal <<< numBlocks, numThreads>>>(d_points, d_steps, p, dir);
        cuQuickSelectShared<16> <<< numBlocks, numThreads>>>(d_points, d_steps, p, dir);
    }
    else if (size > 4)
    {
        cuQuickSelectShared<8> <<< numBlocks, numThreads>>>(d_points, d_steps, p, dir);
    }
    else
    {
        cuQuickSelectShared<4> <<< numBlocks, numThreads>>>(d_points, d_steps, p, dir);
    }
}


void getThreadAndBlockCountForQuickSelect(int n, int p, int &blocks, int &threads)
{
    threads = min(THREADS_PER_BLOCK_QUICK, p);
    blocks = p / threads;
    blocks = min(MAX_BLOCK_DIM_SIZE, blocks);
    blocks = max(1, blocks);
    // printf("block = %d, threads = %d, n = %d, p =%d\n", blocks, threads, n, p );
}

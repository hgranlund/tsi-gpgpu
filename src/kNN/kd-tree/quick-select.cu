#include "quick-select.cuh"
#include <stdio.h>

__device__ void cuPointSwap(struct PointS *p, int a, int b)
{
    struct PointS temp = p[a];
    p[a] = p[b], p[b] = temp;
}

template <int maxStep> __global__
void cuQuickSelectShared(struct PointS *points, int *steps, int p, int dir)
{
    __shared__ struct PointS ss_points[maxStep * THREADS_PER_BLOCK_QUICK];
    struct PointS *s_points = ss_points, *l_points;
    float pivot;
    int pos,
        i,
        list_in_block = p / gridDim.x,
        block_offset = list_in_block * blockIdx.x,
        tid = threadIdx.x,
        rest = p % gridDim.x,
        left,
        right,
        m,
        step_num,
        n;

    if (rest >= gridDim.x - blockIdx.x)
    {
        block_offset += rest - (gridDim.x - blockIdx.x);
        list_in_block++;
    }
    steps += block_offset * 2;

    s_points += (tid * maxStep);

    while ( tid < list_in_block)
    {
        step_num =  tid * 2;
        l_points = points + steps[step_num  ];
        n = steps[step_num   + 1] - steps[step_num  ];
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
void cuQuickSelectGlobal(struct PointS *points, int *steps, int p, int dir)
{
    int pos,
        i,
        list_in_block = p / gridDim.x,
        block_offset = list_in_block * blockIdx.x,
        tid = threadIdx.x,
        rest = p % gridDim.x,
        left,
        right,
        m,
        step_num,
        n;

    struct PointS  *l_points;
    float pivot;
    if (rest >= gridDim.x - blockIdx.x)
    {
        block_offset += rest - (gridDim.x - blockIdx.x);
        list_in_block++;
    }
    steps += block_offset * 2;
    while ( tid < list_in_block)
    {
        step_num = tid * 2;
        l_points = points + steps[step_num];
        n = steps[step_num + 1] - steps[step_num];
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

void quickSelectAndPartition(struct PointS *d_points, int *d_steps, int step , int p, int dir)
{
    int numBlocks, numThreads;
    getThreadAndBlockCountForQuickSelect(step, p, numBlocks, numThreads);
    if (step > 16)
    {
        cuQuickSelectGlobal <<< numBlocks, numThreads>>>(d_points, d_steps, p, dir);
    }
    else if (step > 8)
    {
        if (step * sizeof(PointS) * THREADS_PER_BLOCK_QUICK < MAX_SHARED_MEM)
        {
            cuQuickSelectShared<16> <<< numBlocks, numThreads>>>(d_points, d_steps, p, dir);
        }
        else
        {
            cuQuickSelectGlobal <<< numBlocks, numThreads>>>(d_points, d_steps, p, dir);
        }
    }
    else if (step > 4)
    {
        if (step * sizeof(PointS) * numThreads < MAX_SHARED_MEM)
        {
            cuQuickSelectShared<8> <<< numBlocks, numThreads>>>(d_points, d_steps, p, dir);
        }
        else
        {
            cuQuickSelectGlobal <<< numBlocks, THREADS_PER_BLOCK_QUICK>>>(d_points, d_steps, p, dir);
        }
    }
    else
    {
        if (step * sizeof(PointS) * THREADS_PER_BLOCK_QUICK < MAX_SHARED_MEM)
        {
            cuQuickSelectShared<4> <<< numBlocks, numThreads>>>(d_points, d_steps, p, dir);
        }
        else
        {
            cuQuickSelectGlobal <<< numBlocks, numThreads>>>(d_points, d_steps, p, dir);
        }
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

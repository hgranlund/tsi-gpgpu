#include "quick-select.cuh"
#include <stdio.h>

__device__ void cuPointSwap(struct Point *p, int a, int b)
{
    struct Point temp = p[a];
    p[a] = p[b], p[b] = temp;
}

__device__ void cuCalculateBlockOffsetAndNoOfLists(int n, int &n_per_block, int &block_offset)
{
    int rest = n % gridDim.x;
    n_per_block = n / gridDim.x;
    block_offset = n_per_block * blockIdx.x;

    if (rest >= gridDim.x - blockIdx.x)
    {
        block_offset += rest - (gridDim.x - blockIdx.x);
        n_per_block++;
    }
}

__device__ void cuCopyPoints(struct Point *s_points, struct Point *l_points, int n)
{
    int i;
    for (i = 0; i < n; ++i)
    {
        s_points[i] = l_points[i];
    }
}

template <int max_step, bool in_shared> __global__
void cuQuickSelect(struct Point *points, int *steps, int p, int dir)
{
    __shared__ struct Point ss_points[max_step * THREADS_PER_BLOCK_QUICK];

    struct Point *s_points = ss_points, *l_points;

    float pivot;

    int pos,
        i,
        left,
        right,
        m,
        step_num,
        n,
        list_in_block,
        tid = threadIdx.x,
        block_offset;

    cuCalculateBlockOffsetAndNoOfLists(p, list_in_block, block_offset);

    steps += block_offset * 2;
    s_points += (tid * max_step);

    while ( tid < list_in_block)
    {
        step_num =  tid * 2;
        l_points = points + steps[step_num  ];
        n = steps[step_num   + 1] - steps[step_num  ];
        m = n >> 1;   // same as n/2;
        left = 0;
        right = n - 1;

        if (in_shared)
        {
            cuCopyPoints(s_points, l_points, n);
        }
        else
        {
            s_points = l_points;
        }
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
        if (in_shared)
        {
            cuCopyPoints(l_points, s_points, n);
        }
        tid += blockDim.x;
    }
}

void quickSelectAndPartition(struct Point *d_points, int *d_steps, int step , int p, int dir)
{
    int numBlocks, numThreads;
    getThreadAndBlockCountForQuickSelect(step, p, numBlocks, numThreads);
    if (step > 16)
    {
        cuQuickSelect<1, false> <<< numBlocks, numThreads>>>(d_points, d_steps, p, dir);
    }
    else if (step > 8 && step * sizeof(Point) * numThreads < MAX_SHARED_MEM)
    {
        cuQuickSelect<16, true> <<< numBlocks, numThreads>>>(d_points, d_steps, p, dir);
    }
    else if (step > 4 && step * sizeof(Point) * numThreads < MAX_SHARED_MEM)
    {
        cuQuickSelect<8, true> <<< numBlocks, numThreads>>>(d_points, d_steps, p, dir);
    }
    else if (step * sizeof(Point) * numThreads < MAX_SHARED_MEM)
    {
        cuQuickSelect<4, true> <<< numBlocks, numThreads>>>(d_points, d_steps, p, dir);
    }
    else
    {
        cuQuickSelect<1, false> <<< numBlocks, numThreads>>>(d_points, d_steps, p, dir);
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

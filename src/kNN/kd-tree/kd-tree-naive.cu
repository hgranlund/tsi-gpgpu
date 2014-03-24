#include <kd-tree-naive.cuh>
#include <multiple-radix-select.cuh>
#include <quick-select.cuh>
#include "radix-select.cuh"

#include <stdio.h>
#include <point.h>

#include <helper_cuda.h>

#define debug 0
#include "common-debug.cuh"


__global__
void cuBalanceBranchLeafs(Point* points, int n, int dir)
{
    int
    step = n/gridDim.x,
    blockOffset = step*blockIdx.x,
    tid = threadIdx.x;
    step=step/2;
    Point point1;
    Point point2;
    points += blockOffset;
    while(tid < step){
        point1 = points[tid*2];
        point2 = points[tid*2+1];
        if (point1.p[dir]>point2.p[dir])
        {
            points[tid*2] = point2;
            points[tid*2+1] = point1;
        }
        tid += blockDim.x;
    }
}

void build_kd_tree(Point *h_points, int n)
{


    Point *d_points, *d_swap;
    int p, i, j, numBlocks, numThreads, step;
    int *d_partition;

    checkCudaErrors(
        cudaMalloc(&d_partition, n*sizeof(int)));

    checkCudaErrors(
        cudaMalloc(&d_points, n*sizeof(Point)));

    checkCudaErrors(
        cudaMalloc(&d_swap, n*sizeof(Point)));

    checkCudaErrors(
        cudaMemcpy(d_points, h_points, n*sizeof(Point), cudaMemcpyHostToDevice));

    p = 1;
    step = n/p;
    i = 0;
    while(step >= 8388608 && p <= 2)
    {
        for (j = 0; j <n; j+= n/p)
        {
            radixSelectAndPartition(d_points+j, d_swap+j, d_partition+j, step/2, step, i%3);
        }
        p <<=1;
        step=n/p;
        i++;
    }
    while(step > 256)
    {
        getThreadAndBlockCountMulRadix(n, p, numBlocks, numThreads);
        debugf("n = %d, p = %d, numblosck = %d, numThread =%d\n", n/p, p, numBlocks, numThreads );
        cuBalanceBranch<<<numBlocks,numThreads>>>(d_points, d_swap, d_partition, n/p, p, i%3);
        p <<=1;
        step=n/p;
        i++;
    }
    while(step > 2)
    {
        getThreadAndBlockCountForQuickSelect(n, p, numBlocks, numThreads);
        debugf("n = %d, p = %d, numblosck = %d, numThread =%d\n", n/p, p, numBlocks, numThreads );
        if (step > 16)
        {
            cuQuickSelectGlobal<<<numBlocks,numThreads>>>(d_points, n/p, p, i%3);
        }
        else if (step > 8)
        {
            if (16* sizeof(Point) * numThreads < MAX_SHARED_MEM)
            {
                quickSelectShared(d_points, n/p, p, i%3, 16, numBlocks,numThreads);
            }
            else
            {
                cuQuickSelectGlobal<<<numBlocks,numThreads>>>(d_points, n/p, p, i%3);
            }
        }
        else if (step > 4)
        {
            if (8* sizeof(Point) * numThreads < MAX_SHARED_MEM)
            {
                quickSelectShared(d_points, n/p, p, i%3, 8, numBlocks,numThreads);
            }
            else
            {
                cuQuickSelectGlobal<<<numBlocks,numThreads>>>(d_points, n/p, p, i%3);
            }
        }
        else
        {
            if (4* sizeof(Point) * numThreads < MAX_SHARED_MEM)
            {
                quickSelectShared(d_points, n/p, p, i%3, 4, numBlocks,numThreads);
            }
            else
            {
                cuQuickSelectGlobal<<<numBlocks,numThreads>>>(d_points, n/p, p, i%3);
            }
        }
        p <<=1;
        step=n/p;
        i++;
    }

    numThreads = min(n, THREADS_PER_BLOCK/2);
    numBlocks = n/numThreads;
    numBlocks = min(numBlocks, MAX_BLOCK_DIM_SIZE);
    debugf("n = %d, p = %d, numblosck = %d, numThread =%d\n", n/p, p, numBlocks, numThreads );
    cuBalanceBranchLeafs<<<numBlocks, numThreads>>>(d_points, n, i%3);

    checkCudaErrors(
        cudaMemcpy(h_points, d_points, n*sizeof(Point), cudaMemcpyDeviceToHost));


    checkCudaErrors(cudaFree(d_points));
    checkCudaErrors(cudaFree(d_swap));
    checkCudaErrors(cudaFree(d_partition));
}



#ifndef _COMMON_DEBUG_
#define _COMMON_DEBUG_




#include "stdio.h"
#include "../kNN/point.h"
#include "../kNN/data_types.h"
#include <string.h>
#include <cuda.h>
#include <inc/helper_functions.h>

#define checkCudaErrors(val)           check ( (val), #val, __FILE__, __LINE__ )
#define FILE (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define debugf(fmt, ...) if(debug)printf("%s:%d: " fmt, FILE, __LINE__, __VA_ARGS__);


void printFloatArray_(float *l, int n)
{
    int i;
    if (debug)
    {
        printf("[%3.1f", l[0] );
        for (i = 1; i < n; ++i)
        {
            printf(", %3.1f", l[i] );
        }
        printf("]\n");
    }
}

void printIntArray_(int *l, int n)
{
    int i;
    if (debug)
    {
        printf("[%d", l[0] );
        for (i = 1; i < n; ++i)
        {
            printf(", %d", l[i] );
        }
        printf("]\n");
    }
}


__device__ void printPointsInt_(int *l, int n, char *s)
{
    if (debug)
    {
#if __CUDA_ARCH__>=200
        if (threadIdx.x == 0)
        {
            printf("%10s: [ ", s);
            for (int i = 0; i < n; ++i)
            {
                printf("%3d, ", l[i]);
            }
            printf("]\n");
        }
        __syncthreads();
#endif
    }
}


void h_print_matrix_(Point *points, int n)
{
    if (debug)
    {
        printf("#################\n");
        for (int i = 0; i < n; ++i)
        {
            printf("i = %2d:   ", i );
            for (int j = 0; j < 3; ++j)
            {
                printf(  " %5.0f ", points[i].p[j]);
            }
            printf("\n");
        }
    }
}



__device__
void printPointsMatrix(Point *points, int n, int offset)
{
#if __CUDA_ARCH__>=200
    if (debug)
    {
        __syncthreads();
        if (threadIdx.x == 0 && blockIdx.x == 0)
        {
            printf("####################\n");
            printf("block = %d, blockOffset = %d\n", blockIdx.x, offset );
            for (int i = 0; i < n; ++i)
            {
                printf("i = %3d", i );
                for (int j = 0; j < 3; ++j)
                {
                    printf(  " %3.1f ", points[i].p[j]);
                }
                printf("\n", 1 );
            }
            printf("\n####################\n");
        }
        __syncthreads();
    }
#endif
}

__device__ void d_printPointsArray(Point *l, int n, char *s, int l_debug = 0)
{
    if (debug || l_debug)
    {
#if __CUDA_ARCH__>=200
        if (threadIdx.x == 0)
        {
            printf("%10s: [ ", s);
            for (int i = 0; i < n; ++i)
            {
                printf("%3.1f, ", l[i].p[0]);
            }
            printf("]\n");
        }
        __syncthreads();
#endif
    }
}
__host__  void h_printPointsArray(Point *l, int n, char *s, int l_debug = 0)
{
    if (debug || l_debug)
    {
        printf("%10s: [ ", s);
        for (int i = 0; i < n; ++i)
        {
            printf("%3.1f, ", l[i].p[0]);
        }
        printf("]\n");
    }
}



void printDistArray_(Distance *l, int n)
{
    int i;
    if (debug)
    {
        printf("[(%d - %3.1f)", l[0].index, l[0].value );
        for (i = 1; i < n; ++i)
        {
            printf(", (%d - %3.1f)", l[i].index, l[i].value );
        }
        printf("]\n");
    }
}

#endif

#ifndef _COMMON_KERNELS_
#define _COMMON_KERNELS_


__device__ __host__
unsigned int nextPowTwo(unsigned int x)
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

__device__ __host__
bool isPowTwo(unsigned int x)
{
    return ((x & (x - 1)) == 0);
}

__device__ __host__
unsigned int prevPowTwo(unsigned int n)
{
    if (isPowTwo(n))
    {
        return n;
    }
    n = nextPowTwo(n);
    return n >>= 1;
}

#endif

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
  return ((x&(x-1))==0);
}

__device__ __host__
unsigned int prevPowTwo(unsigned int n)
{
    if (isPowTwo(n))
    {
        return n;
    }
    n = nextPowTwo(n);
    return n >>=1;
}


__device__ int cuSumReduce(int *list, int n)
{
  int half = n/2;
  int tid = threadIdx.x;
  while(tid<half && half > 0)
  {
    list[tid] += list[tid+half];
    half = half/2;
}
return list[0];
}

//TODO must be imporved
__device__  void cuAccumulateIndex(int *list, int n)
{
    if (threadIdx.x == 0)
    {
        int sum=0;
        list[n]=list[n-1];
        int temp=0;
        for (int i = 0; i < n; ++i)
        {
            temp = list[i];
            list[i] = sum;
            sum += temp;
        }
        list[n]+=list[n-1];
    }
}

#endif

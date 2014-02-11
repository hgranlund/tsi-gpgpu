#include "reduction.cuh"
#include "cuda.h"
#include "stdio.h"

// #define SHARED_SIZE_LIMIT 8U
#define SHARED_SIZE_LIMIT 512U

// Can be optimized
__device__ int nearestPowerOf2 (int n)
{
  if (!n){
   return n;  //(0 == 2^0)
 }
 int x = 1;
 while(x <= n)
 {
  x <<= 1;
}
return x;
}

void compare(float &distA, int &indA, float &distB, int &indB, int dir)
{
  float f;
  int i;
  if ((distA  >= distB) == dir)
  {
    f = distA;
    distA  = distB;
    distB = f;
    i = indA;
    indA = indB;
    indB = i;
  }
}

__device__ void cuCompare_r(float &distA, int &indA, float &distB, int &indB, int dir)
{
  float f;
  int i;
  if ((distA  >= distB) == dir)
  {
    f = distA;
    distA  = distB;
    distB = f;
    i = indA;
    indA = indB;
    indB = i;
  }
}



__global__ void min_reduction(float *list,int *ind, int n, int threadOffset)
{


  int  thread1, halfPoint, index1,index2,offset;
  int threadOffset1 = max(1, threadOffset);
  int elements_in_block = nearestPowerOf2(n);
  offset = elements_in_block-n;
  list += blockIdx.x*n;
  ind += blockIdx.x*n;
  while(elements_in_block > 1)
  {
    thread1 = threadIdx.x;
    halfPoint = (elements_in_block / 2);
    while(thread1 < halfPoint)
    {
     if (thread1 + halfPoint   < elements_in_block-offset)
     {
      index1 =thread1 *threadOffset1;
      index2 = index1  + halfPoint * threadOffset1;
      cuCompare_r(list[index1], ind[index1], list[index2], ind[index2], 1);
      }
    thread1 +=blockDim.x;
    }
    __syncthreads();
    offset = 0;
    elements_in_block = halfPoint;
  }
}

void knn_min_reduce(float* dist_dev, int* ind_dev, int n){
  int blockCount, threadCount, elements_in_block, elements_out_of_block, offset;
  blockCount = ceil((float)n/SHARED_SIZE_LIMIT );
  elements_in_block = n/blockCount;
  if (blockCount == 0)
  {
    elements_in_block = n;
    blockCount = 1;
  }
  threadCount = elements_in_block /2;
  threadCount = min(SHARED_SIZE_LIMIT, threadCount);
  elements_out_of_block = n - blockCount * elements_in_block;
  if (elements_out_of_block > 0 )
  {
    offset = n - (elements_out_of_block *2);
    min_reduction<<<1,elements_out_of_block>>>(dist_dev+offset,ind_dev+offset,elements_out_of_block*2,0);
  }
  min_reduction<<<blockCount,threadCount>>>(dist_dev,ind_dev,elements_in_block,0);
  min_reduction<<<1,blockCount>>>(dist_dev,ind_dev,blockCount,elements_in_block);
}



void min_reduce(float *h_list, int *h_ind, int n){

  float *d_list;
  int *d_ind;
  int blockCount, threadCount, elements_in_block;



  blockCount = ceil((float)n/SHARED_SIZE_LIMIT );
  elements_in_block = n/blockCount;
  if (blockCount == 0)
  {
    elements_in_block = n;
    blockCount = 1;
  }
  threadCount = elements_in_block  /2;
  threadCount = min(SHARED_SIZE_LIMIT, threadCount);

  for (int i = n-1; i >= elements_in_block * blockCount; --i)
  {
    compare(h_list[0], h_ind[0], h_list[i], h_ind[i], 1);

  }
  cudaMalloc( (void **) &d_list, n* sizeof(float));
  cudaMalloc( (void **) &d_ind, n* sizeof(int));

  cudaMemcpy(d_list,h_list, n*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_ind,h_ind, n*sizeof(int), cudaMemcpyHostToDevice);
  min_reduction<<<blockCount,threadCount>>>(d_list,d_ind,elements_in_block,0);
  min_reduction<<<1,blockCount>>>(d_list,d_ind,blockCount,elements_in_block);


  cudaMemcpy(h_list,d_list, n*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_ind,d_ind, n*sizeof(int), cudaMemcpyDeviceToHost);

  cudaFree(d_list);
  cudaFree(d_ind);
}

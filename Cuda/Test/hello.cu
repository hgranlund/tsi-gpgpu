#include "stdio.h"

#define N 10

__global__ void add(int *a, int *b, int *c)
{
  int tID = blockIdx.x;
  if (tID<N)
    {
      c[tID] = a[tID] + b[tID];
    }
}
int main()
{
  int a[N], b[N], c[N];
  int *d_a, *d_b, *d_c;


  cudaMalloc((void **) &d_a, N*sizeof(int));
  cudaMalloc((void **) &d_b, N*sizeof(int));
  cudaMalloc((void **) &d_c, N*sizeof(int));

  for (int i = 0; i < N; i++)
  {
    a[i] = i,
    b[i] = 1;
  }
cudaMemcpy(d_a, a, N*sizeof(int), cudaMemcpyHostToDevice);
cudaMemcpy(d_b, b, N*sizeof(int), cudaMemcpyHostToDevice);
cudaMemcpy(d_c, c, N*sizeof(int), cudaMemcpyHostToDevice);

  add<<<N,1>>> (d_a, d_b, d_c);

cudaMemcpy(c,d_c, N*sizeof(int), cudaMemcpyDeviceToHost);

  for (int i = 0; i < N; i++)
  {
    printf("%d + %d = %d\n", a[i], b[i], c[i]);
  }
  return 0;
}

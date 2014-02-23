#ifndef _KERNELS_H_
#define _KERNELS_H_


struct Distance {
 unsigned int index;
 float value;

 __host__  __device__  Distance& operator=(volatile Distance& a)
 {
  index=a.index;
  value=a.value;
  return *this;
}

__host__  __device__  volatile Distance& operator=(Distance& a)
{
  index=a.index;
  value=a.value;
  return *this;
}

__host__ __device__ volatile Distance& operator=(volatile Distance& a) volatile
{
  index=a.index;
  value=a.value;
  return *this;
}


};

void knn_brute_force_garcia(float* ref_host, int ref_width, float* query_host, int query_width, int height, int k, float* dist_host, int* ind_host);
void knn_brute_force(float* ref_host, int ref_nb, float* query_host, int dim, int k, float* dist_host, int* ind_host);


#endif //  _KERNELS_H_

#ifndef _REDUCTION_MOD
#define _REDUCTION_MOD


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

void dist_min_reduce(Distance* dist_dev, unsigned int n);

#endif


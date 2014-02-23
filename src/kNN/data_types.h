#ifndef _DATA_TYPES_
#define _DATA_TYPES_



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

#endif //  _DATA_TYPES_

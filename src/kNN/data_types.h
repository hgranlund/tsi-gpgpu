#ifndef _DATA_TYPES_
#define _DATA_TYPES_

struct Distance
{
    int index;
    float value;

    __device__ __host__  Distance &operator=(volatile Distance &a)
    {
        index = a.index;
        value = a.value;
        return *this;
    }

    __device__ __host__  volatile Distance &operator=(Distance &a)
    {
        index = a.index;
        value = a.value;
        return *this;
    }

    __device__ __host__ volatile Distance &operator=(volatile Distance &a) volatile
    {
        index = a.index;
        value = a.value;
        return *this;
    }
};

#endif //  _DATA_TYPES_

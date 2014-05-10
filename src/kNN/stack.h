#ifndef _STACK_
#define _STACK_

// Collection of structs for internal stack use

// Used to handle the points when in the kd search stack.
struct SPoint
{
    int index;
    int dim;
    float dx;
    int other;
};

// Used to handle the points when they are potential k nearest points.
struct KPoint
{
    int index;
    float dist;
};

#endif //  _DATA_TYPES_

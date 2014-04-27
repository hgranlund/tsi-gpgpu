#ifndef _POINT_
#define _POINT_

struct Point
{
    float p[3];
    int left;
    int right;
};

// Small point used in build kd-tree
struct PointS
{
    float p[3];
};

struct SPoint
{
    int index;
    int dim;
};

struct KPoint
{
    int index;
    float dist;
};

#endif //  _DATA_TYPES_

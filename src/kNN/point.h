#ifndef _POINT_
#define _POINT_

// Main point struct used in finished kd-tree
struct Point
{
    float p[3];
    int left;
    int right;
};

// Small point used as in-data to the kd-tree building algorithm.
struct PointS
{
    float p[3];
};

// Used to handle the points when in the kd search stack.
struct SPoint
{
    int index;
    int dim;
};

// Used to handle the points when they are potential k nearest points.
struct KPoint
{
    int index;
    float dist;
};

#endif //  _DATA_TYPES_

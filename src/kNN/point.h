#ifndef _POINT_
#define _POINT_

// Main point struct used in finished kd-tree
struct Node
{
    float p[3];
    int left;
    int right;
};

// Small point used as in-data to the kd-tree building algorithm.
struct Point
{
    float p[3];
};

#endif //  _DATA_TYPES_

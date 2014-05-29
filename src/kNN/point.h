#ifndef _POINT_
#define _POINT_

// #define ADD_POINT_ID

// Main point struct used in finished kd-tree
struct Node
{
    float p[3];
    int left;
    int right;
# ifdef ADD_POINT_ID
    int id;
#endif
};

// Small point used as in-data to the kd-tree building algorithm.
struct Point
{
    float p[3];
# ifdef ADD_POINT_ID
    int id;
#endif
};
#endif //  _DATA_TYPES_

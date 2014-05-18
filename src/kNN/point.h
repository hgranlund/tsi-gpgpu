#ifndef _POINT_
#define _POINT_

#define KEEP_POINT_INDEX true

// Main point struct used in finished kd-tree
struct Node
{
    float p[3];
    int left;
    int right;
# if KEEP_POINT_INDEX
    int index_orig;
#endif
};

// Small point used as in-data to the kd-tree building algorithm.
struct Point
{
    float p[3];
# if KEEP_POINT_INDEX
    int index_orig;
#endif
};
#endif //  _DATA_TYPES_

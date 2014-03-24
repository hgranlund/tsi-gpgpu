#include <stdlib.h>
#include <math.h>

#include <kd-search.cuh>

int mid(int lower, int upper)
{
    return (int) floor((upper - lower) / 2) + lower;
}

int store_locations(Point *tree, int lower, int upper, int n)
{
    int r;

    if (lower >= upper)
    {
        return -1;
    }

    r = mid(lower, upper);

    tree[r].left = store_locations(tree, lower, r, n);
    tree[r].right = store_locations(tree, r + 1, upper, n);

    return r;
}

float dist(float *qp, Point *points, int x)
{
    float dx = qp[0] - points[x].p[0],
        dy = qp[1] - points[x].p[1],
        dz = qp[2] - points[x].p[2];

    return dx*dx + dy*dy + dz*dz;
}

int nn(float *qp, Point *tree, int dim, int index)
{
    if (tree[index].left == -1 && tree[index].right == -1)
    {
        return index;
    }

    int target, other, d = dim % 3,
    
        target_index = tree[index].right,
        other_index = tree[index].left;
    
    dim++;

    if (tree[index].p[d] > qp[d] || target_index == -1)
    {
        int temp = target_index;

        target_index = other_index;
        other_index = temp;
    }

    target = nn(qp, tree, dim, target_index);
    float target_dist = dist(qp, tree, target);
    float current_dist = dist(qp, tree, index);

    if (current_dist < target_dist)
    {
        target_dist = current_dist;
        target = index;
    }

    if ((tree[index].p[d] - qp[d])*(tree[index].p[d] - qp[d]) > target_dist || other_index == -1)
    {
        return target;
    }

    other = nn(qp, tree, dim, other_index);
    float other_distance = dist(qp, tree, other);

    if (other_distance > target_dist)
    {
        return target;
    }
    return other;
}

// int nn(float *qp, struct Point *tree, float *dists, int dim, int index)
// {
//     float current_dist = dist(qp, tree, index);
//     dists[index] = current_dist;

//     if (tree[index].left == -1 && tree[index].right == -1)
//     {
//         return index;
//     }

//     int target, other, d = dim % 3,
    
//         target_index = tree[index].right,
//         other_index = tree[index].left;
    
//     dim++;

//     if (tree[index].p[d] > qp[d] || target_index == -1)
//     {
//         int temp = target_index;

//         target_index = other_index;
//         other_index = temp;
//     }

//     target = nn(qp, tree, dists, dim, target_index);
//     float target_dist = dists[target];

//     if (current_dist < target_dist)
//     {
//         target_dist = current_dist;
//         target = index;
//     }

//     if ((tree[index].p[d] - qp[d])*(tree[index].p[d] - qp[d]) > target_dist || other_index == -1)
//     {
//         return target;
//     }

//     other = nn(qp, tree, dists, dim, other_index);
//     float other_distance = dists[other];

//     if (other_distance > target_dist)
//     {
//         return target;
//     }
//     return other;
// }
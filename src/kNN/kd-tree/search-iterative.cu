#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <search-iterative.cuh>

int cashe_indexes(Point *tree, int lower, int upper, int n)
{
    int r;

    if (lower >= upper)
    {
        return -1;
    }

    r = (int) floor((float)(upper - lower) / 2) + lower;

    tree[r].left = cashe_indexes(tree, lower, r, n);
    tree[r].right = cashe_indexes(tree, r + 1, upper, n);

    return r;
}

float dist(float *qp, Point *points, int x)
{
    float dx = qp[0] - points[x].p[0],
          dy = qp[1] - points[x].p[1],
          dz = qp[2] - points[x].p[2];

    return dx * dx + dy * dy + dz * dz;
}

void push(int *stack, int *eos, int value)
{
    (*eos)++;
    stack[*eos] = value;
}

int pop(int *stack, int *eos)
{
    if (*eos > -1)
    {
        (*eos)--;
        return stack[*eos + 1];
    } else {
        return -1;
    }
}

int dfs(Point *tree, int n)
{
    int eos = -1,
        *stack = (int *) malloc(n * sizeof stack),

        current,
        target,
        other;

    push(stack, &eos, floor((float)n / 2));

    while(eos > -1)
    {
        current = pop(stack, &eos);

        printf("(%3.1f, %3.1f, %3.1f)\n", tree[current].p[0], tree[current].p[1], tree[current].p[2]);

        target = tree[current].right;
        other = tree[current].left;

        // printf("Current: %d, Target: %d, Other: %d, EOS: %d\n", current, target, other, eos);

        if (target > -1)
        {
            push(stack, &eos, target);
        }
        if (other > -1)
        {
            push(stack, &eos, other);
        }
    }

    return 0;
}

int query_k(float *qp, Point *tree, int dim, int index)
{
    if (tree[index].left == -1 && tree[index].right == -1)
    {
        return index;
    }

    int target,
        other,
        d = dim % 3,

        target_index = tree[index].right,
        other_index = tree[index].left;

    dim++;

    if (tree[index].p[d] > qp[d] || target_index == -1)
    {
        int temp = target_index;

        target_index = other_index;
        other_index = temp;
    }

    target = query_k(qp, tree, dim, target_index);
    float target_dist = dist(qp, tree, target);
    float current_dist = dist(qp, tree, index);

    if (current_dist < target_dist)
    {
        target_dist = current_dist;
        target = index;
    }

    if ((tree[index].p[d] - qp[d]) * (tree[index].p[d] - qp[d]) > target_dist || other_index == -1)
    {
        return target;
    }

    other = query_k(qp, tree, dim, other_index);
    float other_distance = dist(qp, tree, other);

    if (other_distance > target_dist)
    {
        return target;
    }
    return other;
}

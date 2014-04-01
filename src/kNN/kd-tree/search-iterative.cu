#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#include <search-iterative.cuh>

int cashe_indexes(Point *tree, int lower, int upper, int n)
{
    int r;

    if (lower >= upper)
    {
        return -1;
    }

    r = (int) floor((upper - lower) / 2) + lower;

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
    }
    else
    {
        return -1;
    }
}

int peek(int *stack, int eos)
{
    if (eos > -1)
    {
        return stack[eos];
    }
    else
    {
        return -1;
    }
}

int find(int *stack, int eos, int value)
{
    int i;
    for (i = 0; i <= eos; ++i)
    {
        if (stack[i] == value)
        {
            return i;
        }
    }
    return -1;
}

void upDim(int *dim)
{
    if (*dim >= 2)
    {
        (*dim) = 0;
    }
    else
    {
        (*dim)++;
    }
}

void downDim(int *dim)
{
    if (*dim <= 0)
    {
        (*dim) = 2;
    }
    else
    {
        (*dim)--;
    }
}

int dfs(Point *tree, int n)
{
    int eos = -1,
        *stack = (int *) malloc(n * sizeof stack),

         final_visit = 0,

         previous = -1,
         current,
         right,
         left;

    push(stack, &eos, (int) floor(n / 2));

    while (eos > -1)
    {
        // We use a prev variable to keep track of the previously-traversed node.
        // Let’s assume curr is the current node that’s on top of the stack.
        // When prev is curr‘s parent, we are traversing down the tree.
        // In this case, we try to traverse to curr‘s left child if available (ie, push left child to the stack).
        // If it is not available, we look at curr‘s right child.
        // If both left and right child do not exist (ie, curr is a leaf node), we print curr‘s value and pop it off the stack.

        // If prev is curr‘s left child, we are traversing up the tree from the left.
        // We look at curr‘s right child. If it is available, then traverse down the right child (ie, push right child to the stack),
        // otherwise print curr‘s value and pop it off the stack.

        // If prev is curr‘s right child, we are traversing up the tree from the right.
        // In this case, we print curr‘s value and pop it off the stack.

        current = peek(stack, eos);
        left = tree[current].left;
        right = tree[current].right;

        if (previous == left)
        {
            if (right > -1)
            {
                push(stack, &eos, right);
            }
            else
            {
                final_visit = 1;
            }
        }
        else if (previous == right)
        {
            final_visit = 1;
        }
        else
        {
            if (left > -1)
            {
                push(stack, &eos, left);
            }
            else if (right > -1)
            {
                push(stack, &eos, right);
            }
            else
            {
                final_visit = 1;
            }
        }

        if (final_visit)
        {
            current = pop(stack, &eos);
            printf("Current: (%3.1f, %3.1f, %3.1f)\n", tree[current].p[0], tree[current].p[1], tree[current].p[2]);

            final_visit = 0;
        }

        previous = current;
    }

    return 0;
}

// int dfs(Point *tree, int n)
// {
//     int eos = -1,
//         *stack = (int *) malloc(n * sizeof stack),

//         v_eos = -1,
//         *visited = (int *) malloc(n * sizeof visited),

//         current,
//         target,
//         other;

//     push(stack, &eos, floor(n / 2));

//     while(eos > -1)
//     {
//         current = peek(stack, eos);
//         other = tree[current].left;

//         if (other > -1 && find(visited, v_eos, other) == -1)
//         {
//             push(stack, &eos, other);
//         }
//         else
//         {
//             current = pop(stack, &eos);
//             printf("Current: (%3.1f, %3.1f, %3.1f)\n", tree[current].p[0], tree[current].p[1], tree[current].p[2]);

//             push(visited, &v_eos, current);

//             target = tree[current].right;

//             if (target > -1)
//             {
//                 push(stack, &eos, target);
//             }
//         }
//     }

//     return 0;
// }

int query_a(float *qp, Point *tree, int n)
{
    int eos = -1,
        *stack = (int *) malloc(n * sizeof stack),

         final_visit = 0,
         dim = 0,
         best,

         previous = -1,
         current,
         target,
         other;

    float best_dist = FLT_MAX,
          current_dist;

    push(stack, &eos, (int) floor(n / 2));
    upDim(&dim);

    while (eos > -1)
    {
        current = peek(stack, eos);
        target = tree[current].left;
        other = tree[current].right;

        current_dist = dist(qp, tree, current);

        if (current_dist < best_dist)
        {
            best_dist = current_dist;
            best = current;
        }

        if (qp[dim] > tree[current].p[dim])
        {
            int temp = target;

            target = other;
            other = temp;
        }

        if (previous == target)
        {
            if (other > -1 && (tree[current].p[dim] - qp[dim]) * (tree[current].p[dim] - qp[dim]) < best_dist)
            {
                push(stack, &eos, other);
                upDim(&dim);
            }
            else
            {
                final_visit = 1;
            }
        }
        else if (previous == other)
        {
            final_visit = 1;
        }
        else
        {
            if (target > -1)
            {
                push(stack, &eos, target);
                upDim(&dim);
            }
            else if (other > -1)
            {
                push(stack, &eos, other);
                upDim(&dim);
            }
            else
            {
                final_visit = 1;
            }
        }

        if (final_visit)
        {
            current = pop(stack, &eos);
            downDim(&dim);
            // printf("Current: (%3.1f, %3.1f, %3.1f) - dim: %d\n", tree[current].p[0], tree[current].p[1], tree[current].p[2], dim);

            final_visit = 0;
        }

        previous = current;
    }

    return best;
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

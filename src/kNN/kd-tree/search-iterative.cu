#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#include <search-iterative.cuh>

float dist(struct Point qp, struct Point point)
{
    float dx = qp.p[0] - point.p[0],
          dy = qp.p[1] - point.p[1],
          dz = qp.p[2] - point.p[2];

    return dx * dx + dy * dy + dz * dz;
}

void push(int **stackPtr, int value)
{
    *((*stackPtr)++) = value;
}

int pop(int **stackPtr)
{
    return *(--(*stackPtr));
}

int peek(int *stackPtr)
{
    return *(stackPtr - 1);
}

bool isEmpty(int *stackPtr)
{
    return peek(stackPtr) == -1;
}

void initStack(int *stack, int **stackPtr)
{
    *stackPtr = stack;
    push(stackPtr, -1);
}

int target(Point qp, Point current, int dim)
{
    if (qp.p[dim] <= current.p[dim])
    {
        return current.left;
    }
    return current.right;
}

int other(Point qp, Point current, int dim)
{
    if (qp.p[dim] <= current.p[dim])
    {
        return current.right;
    }
    return current.left;
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
    *dim = (*dim + 1) % 3;
}



int query_a(struct Point qp, struct Point *tree, int n)
{
    int stack[50],
        *stackPtr,
        dims[50],
        *dimsPtr,
        dim = 2,
        best = -1,
        current = n / 2;

    float best_dist = FLT_MAX,
          current_dist;

    initStack(stack, &stackPtr);
    initStack(dims, &dimsPtr);

    while (!isEmpty(stackPtr) || current != -1)
    {
        if (current == -1 && !isEmpty(stackPtr))
        {
            current = pop(&stackPtr);
            dim = pop(&dimsPtr);
            printf("(%3.1f, %3.1f, %3.1f) "
                   "current = %d ",
                   tree[current].p[0], tree[current].p[1], tree[current].p[2],
                   current);
            printf("dim = %d\n", dim );

            current_dist = dist(qp, tree[current]);
            if (best_dist > current_dist)
            {
                best_dist = current_dist;
                best = current;
            }
            if ((tree[current].p[dim] - qp.p[dim]) * (tree[current].p[dim] - qp.p[dim]) < best_dist)
            {
                current = other(qp, tree[current], dim);
            }
            else
            {
                current = -1;
            }
        }
        else
        {
            upDim(&dim);
            push(&dimsPtr, dim);
            push(&stackPtr, current);
            current = target(qp, tree[current], dim);
        }
    }
    printf("\n");
    return best;
}
// int query_a(float *qp, struct Point *tree, int n)
// {
//     int eos = -1,
//         *stack = (int *) malloc(n * sizeof stack),

//          final_visit = 0,
//          dim = 0,
//          best,

//          previous = -1,
//          current,
//          target,
//          other;

//     float best_dist = FLT_MAX,
//           current_dist;

//     push(stack, &eos, (int) (n / 2));
//     upDim(&dim);

//     while (eos > -1)
//     {
//         current = peek(stack, eos);
//         target = tree[current].left;
//         other = tree[current].right;

//         current_dist = dist(qp, tree, current);

//         if (current_dist < best_dist)
//         {
//             best_dist = current_dist;
//             best = current;
//         }

//         if (qp[dim] > tree[current].p[dim])
//         {
//             int temp = target;

//             target = other;
//             other = temp;
//         }

//         if (previous == target)
//         {
//             if (other > -1 && (tree[current].p[dim] - qp[dim]) * (tree[current].p[dim] - qp[dim]) < best_dist)
//             {
//                 push(stack, &eos, other);
//                 upDim(&dim);
//             }
//             else
//             {
//                 final_visit = 1;
//             }
//         }
//         else if (previous == other)
//         {
//             final_visit = 1;
//         }
//         else
//         {
//             if (target > -1)
//             {
//                 push(stack, &eos, target);
//                 upDim(&dim);
//             }
//             else if (other > -1)
//             {
//                 push(stack, &eos, other);
//                 upDim(&dim);
//             }
//             else
//             {
//                 final_visit = 1;
//             }
//         }

//         if (final_visit)
//         {
//             current = pop(stack, &eos);
//             downDim(&dim);
//             // printf("Current: (%3.1f, %3.1f, %3.1f) - dim: %d\n", tree[current].p[0], tree[current].p[1], tree[current].p[2], dim);

//             final_visit = 0;
//         }

//         previous = current;
//     }

//     return best;
// }

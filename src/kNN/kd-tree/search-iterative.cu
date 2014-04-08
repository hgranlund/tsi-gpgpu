#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#include <search-iterative.cuh>

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

    push(stack, &eos, (int) (n / 2));
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

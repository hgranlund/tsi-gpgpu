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


void initKStack(KPoint **stack, int n)
{
    (*stack)[0].dist = -1;
    (*stack)++;
    for (int i = 0; i < n; ++i)
    {
        (*stack)[i].dist = FLT_MAX;
    }
}

void insert(struct KPoint *stack, struct KPoint value, int n)
{
    int i = n - 1;
    KPoint swap;
    stack[n - 1] = value;

    while (stack[i].dist < stack[i - 1].dist)
    {
        // printf("dist = %f, old = %f\n", stack[i].dist, stack[i - 1].dist );
        swap = stack[i], stack[i] = stack[i - 1], stack[i - 1] = swap;
        i--;
    }
}


struct KPoint look(struct KPoint *kStack, int n)
{
    return kStack[n - 1];
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

    struct Point current_point;

    initStack(stack, &stackPtr);
    initStack(dims, &dimsPtr);

    while (!isEmpty(stackPtr) || current != -1)
    {
        if (current == -1 && !isEmpty(stackPtr))
        {
            current = pop(&stackPtr);
            current_point = tree[current];
            dim = pop(&dimsPtr);
            printf("(%3.1f, %3.1f, %3.1f) current = %d dim = %d\n",
                   current_point.p[0], current_point.p[1], current_point.p[2], current, dim);

            current_dist = dist(qp, current_point);
            if (best_dist > current_dist)
            {
                best_dist = current_dist;
                best = current;
            }

            current = -1;
            if ((current_point.p[dim] - qp.p[dim]) * (current_point.p[dim] - qp.p[dim]) < best_dist)
            {
                current = other(qp, current_point, dim);
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

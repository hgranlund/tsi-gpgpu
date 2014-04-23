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

void push(int **stack, int value)
{
    *((*stack)++) = value;
}

void initStack(int **stack)
{
    push(stack, -1);
}

int pop(int **stack)
{
    return *(--(*stack));
}

int peek(int *stack)
{
    return *(stack - 1);
}


bool isEmpty(int *stack)
{
    return peek(stack) == -1;
}

void initKStack(KPoint **k_stack, int n)
{
    (*k_stack)[0].dist = -1;
    (*k_stack)++;
    for (int i = 0; i < n; ++i)
    {
        (*k_stack)[i].dist = FLT_MAX;
    }
}

void insert(struct KPoint *k_stack, struct KPoint k_point, int n)
{
    int i = n - 1;
    KPoint swap;
    k_stack[n - 1].index = k_point.index;
    k_stack[n - 1].dist = k_point.dist;

    while (k_stack[i].dist < k_stack[i - 1].dist)
    {
        swap = k_stack[i], k_stack[i] = k_stack[i - 1], k_stack[i - 1] = swap;
        i--;
    }
}

struct KPoint look(struct KPoint *k_stack, int n)
{
    return k_stack[n - 1];
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

void kNN(struct Point qp, struct Point *tree, int n, int k, int *result)
{
    int *stack_ptr = (int *)malloc(51 * sizeof(int)),
         *stack = stack_ptr,
          *d_stack_ptr = (int *)malloc(51 * sizeof(int)),
           *d_stack = d_stack_ptr,
            dim = 2,
            current = n / 2;

    float current_dist;

    struct Point current_point;

    struct KPoint *k_stack_ptr = (struct KPoint *) malloc((k + 1) * sizeof(KPoint)),
                   *k_stack = k_stack_ptr,
                    worst_best;

    worst_best.dist = FLT_MAX;

    initStack(&stack);
    initStack(&d_stack);
    initKStack(&k_stack, k);

    while (!isEmpty(stack) || current != -1)
    {
        if (current == -1 && !isEmpty(stack))
        {
            current = pop(&stack);
            current_point = tree[current];
            dim = pop(&d_stack);

            // printf("(%3.1f, %3.1f, %3.1f) current = %d dim = %d\n",
            // current_point.p[0], current_point.p[1], current_point.p[2], current, dim);

            current_dist = dist(qp, current_point);
            if (worst_best.dist > current_dist)
            {
                worst_best.dist = current_dist;
                worst_best.index = current;
                insert(k_stack, worst_best, k);
                worst_best = look(k_stack, k);
            }

            current = -1;
            if ((current_point.p[dim] - qp.p[dim]) * (current_point.p[dim] - qp.p[dim]) < worst_best.dist)
            {
                current = other(qp, current_point, dim);
            }
        }
        else
        {
            upDim(&dim);
            push(&d_stack, dim);
            push(&stack, current);
            current = target(qp, tree[current], dim);
        }
    }

    free(stack_ptr);
    free(d_stack_ptr);
    free(k_stack_ptr);

    for (int i = 0; i < k; ++i)
    {
        result[i] = k_stack[i].index;
    }
}

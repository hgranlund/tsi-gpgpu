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

void push(struct SPoint **stack, struct SPoint value)
{
    *((*stack)++) = value;
}

void initStack(struct SPoint **stack)
{
    struct SPoint temop;
    temop.index = -1;
    temop.dim = -1;
    push(stack, temop);
}

struct SPoint pop(struct SPoint **stack)
{
    return *(--(*stack));
}

struct SPoint peek(struct SPoint *stack)
{
    return *(stack - 1);
}


bool isEmpty(struct SPoint *stack)
{
    return peek(stack).index == -1;
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

void kNN(struct Point qp, struct Point *tree, int n, int k, int *result, int *visited)
{

    int  dim = 2;

    float current_dist, dx, dx2;

    struct Point current_point;

    struct SPoint *stack_ptr = (struct SPoint *)malloc(51 * sizeof(struct SPoint)),
                   *stack = stack_ptr,
                    current;

    struct KPoint *k_stack_ptr = (struct KPoint *) malloc((k + 1) * sizeof(KPoint)),
                   *k_stack = k_stack_ptr,
                    worst_best;

    current.index = n / 2;
    worst_best.dist = FLT_MAX;

    initStack(&stack);
    initKStack(&k_stack, k);

    while (!isEmpty(stack) || current.index != -1)
    {
        if (current.index == -1 && !isEmpty(stack))
        {
            current = pop(&stack);
            current_point = tree[current.index];
            dim = current.dim;

            // printf("(%3.1f, %3.1f, %3.1f) current = %d dim = %d\n",
            //        current_point.p[0], current_point.p[1], current_point.p[2], current, dim);

            current.index = -1; //Lage en cache null current

            dx = current_point.p[dim] - qp.p[dim];
            dx2 = dx * dx;

            if (dx2 < worst_best.dist)
            {
                current.index = other(qp, current_point, dim);
            }

            (*visited)++;
        }
        else
        {
            current_point = tree[current.index];
            current_dist = dist(qp, current_point);

            if (worst_best.dist > current_dist)
            {
                worst_best.dist = current_dist;
                worst_best.index = current.index;
                insert(k_stack, worst_best, k);
                worst_best = look(k_stack, k);
            }

            upDim(&dim);
            current.dim = dim;
            push(&stack, current);

            current.index = target(qp, current_point, dim);
        }
    }

    free(stack_ptr);
    free(k_stack_ptr);

    // printf("\n");

    for (int i = 0; i < k; ++i)
    {
        result[i] = k_stack[i].index;
    }
}

void kNN(struct Point qp, struct Point *tree, int n, int k, int *result)
{
    int visited;
    kNN(qp, tree, n, k, result, &visited);
}

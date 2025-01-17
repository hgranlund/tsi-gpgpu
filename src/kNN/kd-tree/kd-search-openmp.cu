#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

// #ifdef HAVE_OPENMP
#include <omp.h>
// #endif

#include "kd-search-openmp.cuh"



float dist(struct Point qp, struct Node point)
{
    float dx = qp.p[0] - point.p[0],
          dy = qp.p[1] - point.p[1],
          dz = qp.p[2] - point.p[2];

    return (dx * dx) + (dy * dy) + (dz * dz);
}

void initStack(struct SPoint **stack)
{
    (*stack)[0].index = -1;
    (*stack)++;
}

int isEmpty(struct SPoint *stack)
{
    return peek(stack).index == -1;
}

void push(struct SPoint **stack, struct SPoint value)
{
    *((*stack)++) = value;
}

struct SPoint pop(struct SPoint **stack)
{
    return *(--(*stack));
}

struct SPoint peek(struct SPoint *stack)
{
    return *(stack - 1);
}

void initKStack(struct KPoint **k_stack, int n)
{
    (*k_stack)--;
    for (int i = 1; i <= n; ++i)
    {
        (*k_stack)[i].dist = FLT_MAX;
        (*k_stack)[i].index = -1;
    }
}

void insert(struct KPoint *k_stack, struct KPoint k_point, int n)
{
    int i_child, now;
    struct KPoint child, child_tmp_2;
    for (now = 1; now * 2 <= n ; now = i_child)
    {
        i_child = now * 2;
        child = k_stack[i_child];
        child_tmp_2 = k_stack[i_child + 1];
        if (i_child <= n && child_tmp_2.dist > child.dist )
        {
            i_child++;
            child = child_tmp_2;
        }

        if (i_child <= n && k_point.dist < child.dist)
        {
            k_stack[now] = child;
        }
        else
        {
            break;
        }
    }
    k_stack[now] = k_point;
}

struct KPoint look(struct KPoint *k_stack)
{
    return k_stack[1];
}

void upDim(int &dim)
{
    dim = (dim + 1) % 3;
}

int target(struct Point qp, struct Node current, float dx)
{
    if (dx > 0)
    {
        return current.left;
    }
    return current.right;
}

int other(struct Point qp, struct Node current, float dx)
{
    if (dx > 0)
    {
        return current.right;
    }
    return current.left;
}

void kNN(struct Point qp, struct Node *tree, int n, int k, int *result,
         struct SPoint *stack_ptr, struct KPoint *k_stack_ptr)
{
    int  i, dim = 2;
    float current_dist, dx, dx2;

    struct Node current_point;
    struct SPoint *stack = stack_ptr,
                           current;
    struct KPoint *k_stack = k_stack_ptr,
                           worst_best;

    initStack(&stack);
    initKStack(&k_stack, k);

    worst_best = look(k_stack);
    current.index = n / 2;

    while (!isEmpty(stack) || current.index != -1)
{
        if (current.index == -1 && !isEmpty(stack))
        {
            current = pop(&stack);
            dim = current.dim;

            dx = current.dx;
            dx2 = dx * dx;

            current.index = (dx2 < worst_best.dist) ? current.other : -1;
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
                worst_best = look(k_stack);
            }

            upDim(dim);
            current.dim = dim;
            current.dx = current_point.p[dim] - qp.p[dim];
            current.other = other(qp, current_point, current.dx);
            push(&stack, current);

            current.index = target(qp, current_point, current.dx);
        }
    }

    k_stack++;
    for (i = 0; i < k; ++i)
    {
        result[i] = k_stack[i].index;
    }
}

void mpQueryAll(struct Point *query_points, struct Node *tree, int n_qp, int n_tree, int k, int *result)
{
    int stack_size = log2((float)n_tree) + 5;
    #pragma omp parallel
    {
        int th_id = omp_get_thread_num();
        struct SPoint *stack_ptr = (struct SPoint *) malloc(stack_size * sizeof(struct SPoint));
        struct KPoint *k_stack_ptr = (struct KPoint *) malloc(k * sizeof(struct KPoint));

        while (th_id < n_qp)
        {
            kNN(query_points[th_id], tree, n_tree, k, result + (th_id * k), stack_ptr, k_stack_ptr);
            th_id += omp_get_num_threads();
        }

        free(stack_ptr);
        free(k_stack_ptr);
    }
}

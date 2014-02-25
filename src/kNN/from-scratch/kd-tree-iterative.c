#include <stdlib.h>
#include <stdio.h>
#include <math.h>

void swap(double *x, int a, int b)
{
    double t = x[a];
    x[a] = x[b], x[b] = t;
}

int midpoint(int lower, int upper)
{
    return (int) floor((upper - lower) / 2) + lower;
}

double quick_select(int k, double *x, int lower, int upper)
{
    int pos, i,
        left = lower, 
        right = upper - 1;

    double pivot;

    while (left < right)
    {
        pivot = x[k];
        swap(x, k, right);
        for (i = pos = left; i < right; i++)
        {
            if (x[i] < pivot)
            {
                swap(x, i, pos);
                pos++;
            }
        }
        swap(x, right, pos);
        if (pos == k) break;
        if (pos < k) left = pos + 1;
        else right = pos - 1;
    }
    return x[k];
}

int center_median(double *x, int lower, int upper)
{
    int i, r = midpoint(lower, upper);

    double median = quick_select(r, x, lower, upper);

    for (i = lower; i < upper; ++i)
    {
        if (x[i] == median)
        {
            swap(x, i, r);
            return;
        }
    }
}

void balance_branch(double *x, int lower, int upper)
{
    if (lower >= upper) return;

    int i, r = midpoint(lower, upper);

    center_median(x, lower, upper);

    upper--;

    for (i = lower; i < r; ++i)
    {
        if (x[i] > x[r])
        {
            while (x[upper] > x[r])
            {
                upper--;
            }
            swap(x, i, upper);
        }
    }

    // To enable direct recusive execution.
    // balance_branch(x, lower, r);
    // balance_branch(x, r + 1, upper);
}

void build_kd_tree(double *x, int len)
{
    int i, j, p, step,
        h = ceil(log2(len + 1) - 1);
    for (i = 0; i < h; ++i)
    {
        p = pow(2, i);
        step = (int) floor(len / p);

        for (j = 0; j < p; ++j)
        {
            balance_branch(x, (1 + step) * j, step * (1 + j));
        }
    }
    return;
}
 
void print_tree(double tree[], int level, int lower, int upper)
{
    if (lower >= upper) return;

    int i, r = midpoint(lower, upper);

    printf("|");
    for (i = 0; i < level; ++i) { printf("--"); }
    printf(" %lf\n", tree[r]);

    print_tree(tree, 1 + level, lower, r);
    print_tree(tree, 1 + level, r + 1, upper);
}

int main(int argc, char *argv[])
{
    double points[] = {15, 14, 13, 12, 11, 10, 9, 7, 8, 6, 5, 4, 3, 2, 1};
    int len = 15;

    printf("initial list:\n");
    print_tree(points, 0, 0, len);
    printf("==================\n");

    printf("kd tree:\n");
    build_kd_tree(points, len);
    print_tree(points, 0, 0, len);
    printf("==================\n");

    return 0;
}

// TODO:
// Thurrow testing, including non-perfect binary trees.
// Add support for three-dimentional points.
// Paralellize all the things.
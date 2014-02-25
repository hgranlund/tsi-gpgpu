#include <stdlib.h>
#include <stdio.h>
#include <math.h>
 
struct node
{
    double x[3];
};

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
    int i,
        r = midpoint(lower, upper);

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

void build_kd_tree(double *x, int lower, int upper)
{
    if (lower >= upper) return;

    int i,
        r = midpoint(lower, upper);

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

    // To enable recusive execution.
    build_kd_tree(x, lower, r);
    build_kd_tree(x, r + 1, upper);
}

void kd_tree(double *x, int lower, int upper)
{
    return;
}
 
void print_tree(double tree[], int level, int lower, int upper)
{
    if (lower >= upper) return;

    int r = midpoint(lower, upper);

    int i;
    printf("|");
    for (i = 0; i < level; ++i) { printf("--"); }
    printf(" %lf\n", tree[r]);

    int next_level = level + 1;
    print_tree(tree, next_level, lower, r);
    print_tree(tree, next_level, r + 1, upper);
}

int main(int argc, char *argv[])
{
    double tree[] = {15, 14, 13, 12, 11, 10, 9, 7, 8, 6, 5, 4, 3, 2, 1};
    int len = 15;

    printf("Tree print test:\n");
    print_tree(tree, 0, 0, len);
    printf("==================\n");

    printf("Tree build test 1:\n");
    build_kd_tree(tree, 0, len);
    print_tree(tree, 0, 0, len);
    printf("==================\n");

    // printf("Tree build test 2:\n");
    // build_kd_tree(tree, 0, 7);
    // print_tree(tree, 0, 0, len);
    // printf("==================\n");

    // printf("Tree build test 3:\n");
    // build_kd_tree(tree, 8, len);
    // print_tree(tree, 0, 0, len);
    // printf("==================\n");

    return 0;
}
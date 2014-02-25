#include <stdlib.h>
#include <stdio.h>
#include <math.h>

void swap(double **x, int a, int b)
{
    double *t = x[a];
    x[a] = x[b], x[b] = t;
}

int midpoint(int lower, int upper)
{
    return (int) floor((upper - lower) / 2) + lower;
}

double quick_select(int k, double **x, int lower, int upper, int dim)
{
    int pos, i,
    left = lower,
    right = upper - 1;

    double pivot;

    while (left < right)
    {
        pivot = x[k][dim];
        swap(x, k, right);
        for (i = pos = left; i < right; i++)
        {
            if (x[i][dim] < pivot)
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
    return x[k][dim];
}

int center_median(double **x, int lower, int upper, int dim)
{
    int i, r = midpoint(lower, upper);

    double median = quick_select(r, x, lower, upper, dim);

    for (i = lower; i < upper; ++i)
    {
        if (x[i][dim] == median)
        {
            swap(x, i, r);
            return;
        }
    }
}

void balance_branch(double **x, int lower, int upper, int dim)
{
    if (lower >= upper) return;

    int i, r = midpoint(lower, upper);

    center_median(x, lower, upper, dim);

    upper--;

    for (i = lower; i < r; ++i)
    {
        if (x[i][dim] > x[r][dim])
        {
            while (x[upper][dim] > x[r][dim])
            {
                upper--;
            }
            swap(x, i, upper);
        }
    }

    // To enable direct recusive execution.
    // balance_branch(x, lower, r, 0);
    // balance_branch(x, r + 1, upper, 0);
}

void build_kd_tree(double **x, int len)
{
    int i, j, p, step,
    h = ceil(log2(len + 1) - 1);
    for (i = 0; i < h; ++i)
    {
        p = pow(2, i);
        step = (int) floor(len / p);

        for (j = 0; j < p; ++j)
        {
            balance_branch(x, (1 + step) * j, step * (1 + j), i%3);
        }
    }
    return;
}

void print_tree(double **tree, int level, int lower, int upper)
{
    if (lower >= upper)
    {
        return;
    }

    int i, r = midpoint(lower, upper);

    printf("|");
    for (i = 0; i < level; ++i)
    {
        printf("--");
    }
    printf("(%lf, %lf, %lf)\n", tree[r][0], tree[r][1], tree[r][2]);

    print_tree(tree, 1 + level, lower, r);
    print_tree(tree, 1 + level, r + 1, upper);
}

int main(int argc, char *argv[])
{


    int i,j,len = 15, wlen = 6;
    double **points, **wiki;
    points = (double**) malloc(len * sizeof(double*));

    for ( i = 0; i < len; ++i)
    {
        points[i] = (double*) malloc(3 * sizeof(double));

        for ( j = 0; j < 3; ++j)
        {
            points[i][j] = len - i*j;
        }
    }

    // (2,3), (5,4), (9,6), (4,7), (8,1), (7,2).
    wiki = (double**) malloc(len * sizeof(double*));
    for ( i = 0; i < wlen; ++i)
    {
        wiki[i] = (double*) malloc(3 * sizeof(double));
    }

    wiki[0][0] = 2, wiki[0][1] = 3, wiki[0][2] = 0;
    wiki[1][0] = 5, wiki[1][1] = 4, wiki[1][2] = 0;
    wiki[2][0] = 9, wiki[2][1] = 6, wiki[2][2] = 0;
    wiki[3][0] = 4, wiki[3][1] = 7, wiki[3][2] = 0;
    wiki[4][0] = 8, wiki[4][1] = 1, wiki[4][2] = 0;
    wiki[5][0] = 7, wiki[5][1] = 2, wiki[5][2] = 0;


    printf("initial list:\n");
    print_tree(points, 0, 0, len);
    printf("==================\n");

    printf("kd tree:\n");
    build_kd_tree(points, len);
    print_tree(points, 0, 0, len);
    printf("==================\n");

    printf("initial wiki:\n");
    print_tree(wiki, 0, 0, wlen);
    printf("==================\n");

    printf("wiki tree:\n");
    build_kd_tree(wiki, wlen);
    print_tree(wiki, 0, 0, wlen);
    printf("==================\n");


    for ( i = 0; i < len; ++i)
    {
        free(points[i]);
    }
    free(points);

    for ( i = 0; i < len; ++i)
    {
        free(wiki[i]);
    }
    free(wiki);
    return 0;
}

// TODO:
// Thurrow testing, including non-perfect binary trees.
// Add support for three-dimentional points.
// Paralellize all the things.

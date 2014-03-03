#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

int ind(int i, int j, int n)
{
    return i + j * n;
}

void swap(float *x, int a, int b, int n)
{
    float tx = x[ind(a, 0, n)],
        ty = x[ind(a, 1, n)],
        tz = x[ind(a, 2, n)];

    x[ind(a, 0, n)] = x[ind(b, 0, n)], x[ind(b, 0, n)] = tx;
    x[ind(a, 1, n)] = x[ind(b, 1, n)], x[ind(b, 1, n)] = ty;
    x[ind(a, 2, n)] = x[ind(b, 2, n)], x[ind(b, 2, n)] = tz;
}

int midpoint(int lower, int upper)
{
    return (int) floor((upper - lower) / 2) + lower;
}

float quick_select(int k, float *x, int lower, int upper, int dim, int n)
{
    int pos, i,
    left = lower,
    right = upper - 1;

    float pivot;

    while (left < right)
    {
        pivot = x[ind(k, dim, n)];
        swap(x, k, right, n);
        for (i = pos = left; i < right; i++)
        {
            if (x[ind(i, dim, n)] < pivot)
            {
                swap(x, i, pos, n);
                pos++;
            }
        }
        swap(x, right, pos, n);
        if (pos == k) break;
        if (pos < k) left = pos + 1;
        else right = pos - 1;
    }
    return x[ind(k, dim, n)];
}

int center_median(float *x, int lower, int upper, int dim, int n)
{
    int i, r = midpoint(lower, upper);

    float median = quick_select(r, x, lower, upper, dim, n);

    for (i = lower; i < upper; ++i)
    {
        if (x[ind(i, dim, n)] == median)
        {
            swap(x, i, r, n);
            return;
        }
    }
}

void balance_branch(float *x, int lower, int upper, int dim, int n)
{
    if (lower >= upper) return;

    int i, r = midpoint(lower, upper);

    center_median(x, lower, upper, dim, n);

    upper--;

    for (i = lower; i < r; ++i)
    {
        if (x[ind(i, dim, n)] > x[ind(r, dim, n)])
        {
            while (x[ind(upper, dim, n)] > x[ind(r, dim, n)])
            {
                upper--;
            }
            swap(x, i, upper, n);
        }
    }

    // To enable direct recusive execution.
    // balance_branch(x, lower, r, 0, n);
    // balance_branch(x, r + 1, upper, 0, n);
}

void build_kd_tree(float *x, int n)
{
    int i, j, p, step,
    h = ceil(log2(n + 1) - 1);
    for (i = 0; i < h; ++i)
    {
        p = pow(2, i);
        step = (int) floor(n / p);

        for (j = 0; j < p; ++j)
        {
            balance_branch(x, (1 + step) * j, step * (1 + j), i % 3, n);
        }
    }
    return;
}

float euclid_distance(float ax, float ay, float az, float bx, float by, float bz)
{
    return sqrt(pow((ax - bx), 2.0) + pow((ay - by), 2.0) + pow((az - bz), 2.0));
}

int closest_point(float *qp, float *tree, int a, int b, int n)
{
    float a_distance = euclid_distance(qp[0], qp[1], qp[2], tree[ind(a, 0, n)], tree[ind(a, 1, n)], tree[ind(a, 2, n)]),
        b_distance = euclid_distance(qp[0], qp[1], qp[2], tree[ind(b, 0, n)], tree[ind(b, 1, n)], tree[ind(b, 2, n)]);

    if (a_distance < b_distance)
    {
        return a;
    }
    return b;
}

// Rewrite for new data structure.
// void nearest(struct kd_node_t *root, struct kd_node_t *nd, int i, int dim, struct kd_node_t **best, double *best_dist)
// {
//     double d, dx, dx2;
 
//     if (!root) return;
//     d = dist(root, nd, dim);
//     dx = root->x[i] - nd->x[i];
//     dx2 = dx * dx;
 
//     visited ++;
 
//     if (!*best || d < *best_dist) {
//         *best_dist = d;
//         *best = root;
//     }
 
//     if (!*best_dist) return;
 
//     if (++i >= dim) i = 0;
 
//     nearest(dx > 0 ? root->left : root->right, nd, i, dim, best, best_dist);
//     if (dx2 >= *best_dist) return;
//     nearest(dx > 0 ? root->right : root->left, nd, i, dim, best, best_dist);
// }

int nearest(float *qp, float *tree, int lower, int upper, int n)
{
    if (lower >= upper)
    {
        return lower;
    }


    int r = midpoint(lower, upper),
        left_best = nearest(qp, tree, lower, r, n),
        right_best = nearest(qp, tree, r + 1, upper, n);

    printf("(%3.1f, %3.1f, %3.1f)\n", tree[ind(r, 0, n)], tree[ind(r, 1, n)], tree[ind(r, 2, n)]);
    printf("r = %d\n", r);

    return closest_point(qp, tree, left_best, right_best, n);
}

void print_tree(float *tree, int level, int lower, int upper, int n)
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
    printf("(%3.1f, %3.1f, %3.1f)\n", tree[ind(r, 0, n)], tree[ind(r, 1, n)], tree[ind(r, 2, n)]);

    print_tree(tree, 1 + level, lower, r, n);
    print_tree(tree, 1 + level, r + 1, upper, n);
}

double WallTime()
{
    struct timeval tmpTime;
    gettimeofday(&tmpTime, NULL);
    return tmpTime.tv_sec + tmpTime.tv_usec/1.0e6;
}

int main(int argc, char *argv[])
{
    int i, j, n = 10, wn = 6, debug = 1;
    float *points, *wiki;

    points = (float*) malloc(n * 3 * sizeof(float));

    srand(time(NULL));
    for ( i = 0; i < n; ++i)
    {
        for ( j = 0; j < 3; ++j)
        {
            points[ind(i, j, n)] = rand() % 1000;
        }
    }

    if (!debug)
    {
        double time = WallTime();
        build_kd_tree(points, n);
        printf("Build duration for %d points: %lf (ms)\n", n, (WallTime() - time) * 1000);
    }

    if (debug)
    {
        wiki = (float*) malloc(wn * 3 * sizeof(float*));

        // (2,3), (5,4), (9,6), (4,7), (8,1), (7,2).
        wiki[ind(0, 0, wn)] = 2, wiki[ind(0, 1, wn)] = 3, wiki[ind(0, 2, wn)] = 0;
        wiki[ind(1, 0, wn)] = 5, wiki[ind(1, 1, wn)] = 4, wiki[ind(1, 2, wn)] = 0;
        wiki[ind(2, 0, wn)] = 9, wiki[ind(2, 1, wn)] = 6, wiki[ind(2, 2, wn)] = 0;
        wiki[ind(3, 0, wn)] = 4, wiki[ind(3, 1, wn)] = 7, wiki[ind(3, 2, wn)] = 0;
        wiki[ind(4, 0, wn)] = 8, wiki[ind(4, 1, wn)] = 1, wiki[ind(4, 2, wn)] = 0;
        wiki[ind(5, 0, wn)] = 7, wiki[ind(5, 1, wn)] = 2, wiki[ind(5, 2, wn)] = 0;

        printf("initial list:\n");
        print_tree(points, 0, 0, n, n);
        printf("==================\n");

        printf("kd tree:\n");
        build_kd_tree(points, n);
        print_tree(points, 0, 0, n, n);
        printf("==================\n");

        printf("initial wiki:\n");
        print_tree(wiki, 0, 0, wn, wn);
        printf("==================\n");

        printf("wiki tree:\n");
        build_kd_tree(wiki, wn);
        print_tree(wiki, 0, 0, wn, wn);
        printf("==================\n");

        float query_point[] = {0, 0, 0};
        int best_fit = nearest(query_point, wiki, 0, wn, wn);
        printf("Closest point to (%3.1f, %3.1f, %3.1f) was (%3.1f, %3.1f, %3.1f) located at %d\n",
            query_point[0], query_point[1], query_point[2],
            wiki[ind(best_fit, 0, wn)], wiki[ind(best_fit, 1, wn)], wiki[ind(best_fit, 2, wn)],
            best_fit);
        printf("==================\n");

        free(wiki);

        printf("euclid_distance test:\n");
        printf("Should be 3: %f\n", euclid_distance(2, 2, 1, 0, 0, 0));
        printf("Should be 3: %f\n", euclid_distance(4, -2, 1, 2, -4, 0));
        printf("==================\n");
    }
    free(points);
    return 0;
}

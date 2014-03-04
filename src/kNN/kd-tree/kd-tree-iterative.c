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

float distance_to_query_point(float *qp, float x, float y, float z)
{
    return (qp[0] - x)*(qp[0] - x) + (qp[1] - y)*(qp[1] - y) + (qp[2] - z)*(qp[2] - z);
}

int nearest(float *qp, float *tree, int lower, int upper, int dim, int n)
{
    if (lower >= upper - 1)
    {
        if (lower >= n)
        {
            return n - 1;
        }
        return lower;
    }

    int target, other,

        r = midpoint(lower, upper),
        d = dim % 3,
    
        target_lower = r + 1,
        target_upper = upper,
        other_lower = lower,
        other_upper = r;
    
    dim++;

    if (tree[ind(r, d, n)] > qp[d])
    {
        target_lower = lower;
        target_upper = r;
        other_lower = r + 1;
        other_upper = upper;
    }

    target = nearest(qp, tree, target_lower, target_upper, dim, n);

    float target_dist = distance_to_query_point(qp, tree[ind(target, 0, n)], tree[ind(target, 1, n)], tree[ind(target, 2, n)]),
        current_dist = distance_to_query_point(qp, tree[ind(r, 0, n)], tree[ind(r, 1, n)], tree[ind(r, 2, n)]);

    if (current_dist < target_dist)
    {
        target_dist = current_dist;
        target = r;
    }

    if ((tree[ind(r, d, n)] - qp[d])*(tree[ind(r, d, n)] - qp[d]) > target_dist)
    {
        return target;
    }

    other = nearest(qp, tree, other_lower, other_upper, dim, n);

    float other_distance = distance_to_query_point(qp, tree[ind(other, 0, n)], tree[ind(other, 1, n)], tree[ind(other, 2, n)]);

    if (other_distance > target_dist)
    {
        return target;
    }
    return other;
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

double wall_time()
{
    struct timeval tmpTime;
    gettimeofday(&tmpTime, NULL);
    return tmpTime.tv_sec + tmpTime.tv_usec/1.0e6;
}

int test_nearest(float *tree, int n, float qx, float qy, float qz, float ex, float ey, float ez)
{
    float query_point[3];
    query_point[0] = qx, query_point[1] = qy, query_point[2] = qz;
    
    int best_fit = nearest(query_point, tree, 0, n, 0, n);

    float actual = tree[ind(best_fit, 0, n)] + tree[ind(best_fit, 1, n)] + tree[ind(best_fit, 2, n)];
    float expected = ex + ey + ez;

    if (actual == expected)
    {
        return 0;
    }

    return 1;

    printf("Closest point to (%3.1f, %3.1f, %3.1f) was (%3.1f, %3.1f, %3.1f) located at %d\n",
        query_point[0], query_point[1], query_point[2],
        tree[ind(best_fit, 0, n)], tree[ind(best_fit, 1, n)], tree[ind(best_fit, 2, n)],
        best_fit);
    printf("==================\n");
}

int main(int argc, char *argv[])
{
    int i, j, n = 1000000, wn = 6, debug = 0;
    float *points, *wiki;

    if (debug)
    {
        n = 10;
    }

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
        double time = wall_time();
        build_kd_tree(points, n);
        printf("Build duration for %d points: %lf (ms)\n", n, (wall_time() - time) * 1000);

        float query_point[3];
        time = wall_time();
        int sum = 0, test_runs = 100000;

        for (i = 0; i < test_runs; i++) {
            query_point[0] = rand() % 1000;
            query_point[1] = rand() % 1000;
            query_point[2] = rand() % 1000;
            nearest(query_point, points, 0, n, 0, n);
        }
        printf("Total time for %d queries: %lf (ms)\n", test_runs, ((wall_time() - time) * 1000));
        printf("Average query duration: %lf (ms)\n", ((wall_time() - time) * 1000) / test_runs);
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

        int not_passed_test = test_nearest(wiki, wn, 2, 3, 0, 2, 3, 0)
            + test_nearest(wiki, wn, 5, 4, 0, 5, 4, 0)
            + test_nearest(wiki, wn, 9, 6, 0, 9, 6, 0)
            + test_nearest(wiki, wn, 4, 7, 0, 4, 7, 0)
            + test_nearest(wiki, wn, 8, 1, 0, 8, 1, 0)
            + test_nearest(wiki, wn, 7, 2, 0, 7, 2, 0)
            + test_nearest(wiki, wn, 10, 10, 0, 9, 6, 0)
            + test_nearest(wiki, wn, 0, 0, 0, 2, 3, 0)
            + test_nearest(wiki, wn, 4, 4, 0, 5, 4, 0)
            + test_nearest(wiki, wn, 3, 2, 0, 2, 3, 0)
            + test_nearest(wiki, wn, 2, 6, 0, 4, 7, 0)
            + test_nearest(wiki, wn, 10, 0, 0, 8, 1, 0)
            + test_nearest(wiki, wn, 0, 10, 0, 4, 7, 0);

        // test_nearest(wiki, wn, 10, 0, 0, 8, 1, 0);

        if (not_passed_test)
        {
            printf("nearest function not working right!\n");
            printf("==================\n");   
        }
        else {
            printf("nearest function still works!\n");
            printf("==================\n");
        }

        free(wiki);
    }
    free(points);
    return 0;
}

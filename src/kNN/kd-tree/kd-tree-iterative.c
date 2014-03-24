#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

struct Point
{
    float p[3];
    int left;
    int right;
};

int ind(int i, int j, int n)
{
    return i + j * n;
}

void swap(struct Point *points, int a, int b)
{
    struct Point t = points[a];
    points[a] = points[b], points[b] = t;
}

int midpoint(int lower, int upper)
{
    return (int) floor((upper - lower) / 2) + lower;
}

float quick_select(int k, struct Point *x, int lower, int upper, int dim)
{
    int pos, i,
    left = lower,
    right = upper - 1;

    float pivot;

    while (left < right)
    {
        pivot = x[k].p[dim];
        swap(x, k, right);
        for (i = pos = left; i < right; i++)
        {
            if (x[i].p[dim] < pivot)
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
    return x[k].p[dim];
}

int center_median(struct Point *x, int lower, int upper, int dim)
{
    int i, r = midpoint(lower, upper);

    float median = quick_select(r, x, lower, upper, dim);

    for (i = lower; i < upper; ++i)
    {
        if (x[i].p[dim] == median)
        {
            swap(x, i, r);
            return;
        }
    }
}

void balance_branch(struct Point *x, int lower, int upper, int dim)
{
    if (lower >= upper) return;

    int i, r = midpoint(lower, upper);

    center_median(x, lower, upper, dim);

    upper--;

    for (i = lower; i < r; ++i)
    {
        if (x[i].p[dim] > x[r].p[dim])
        {
            while (x[upper].p[dim] > x[r].p[dim])
            {
                upper--;
            }
            swap(x, i, upper);
        }
    }

    // To enable direct recursive execution.
    // balance_branch(x, lower, r, 0);
    // balance_branch(x, r + 1, upper, 0);
}

void build_kd_tree(struct Point *x, int n)
{
    int i, j, p, step,
    h = ceil(log2(n + 1) - 1);
    for (i = 0; i < h; ++i)
    {
        p = pow(2, i);
        step = (int) floor(n / p);

        for (j = 0; j < p; ++j)
        {
            balance_branch(x, (1 + step) * j, step * (1 + j), i % 3);
        }
    }
    return;
}

int store_locations(struct Point *tree, int lower, int upper, int n)
{
    int r;

    if (lower >= upper)
    {
        return -1;
    }

    r = midpoint(lower, upper);

    tree[r].left = store_locations(tree, lower, r, n);
    tree[r].right = store_locations(tree, r + 1, upper, n);

    return r;
}

float dist(float *qp, struct Point *points, int x)
{
    float dx = qp[0] - points[x].p[0],
        dy = qp[1] - points[x].p[1],
        dz = qp[2] - points[x].p[2];

    return dx*dx + dy*dy + dz*dz;
}

int nn(float *qp, struct Point *tree, int dim, int index)
{

    if (tree[index].left == -1 && tree[index].right == -1)
    {
        return index;
    }

    int target, other, d = dim % 3,
    
        target_index = tree[index].right,
        other_index = tree[index].left;
    
    dim++;

    if (tree[index].p[d] > qp[d] || target_index == -1)
    {
        int temp = target_index;

        target_index = other_index;
        other_index = temp;
    }

    target = nn(qp, tree, dim, target_index);
    float target_dist = dist(qp, tree, target);
    float current_dist = dist(qp, tree, index);

    if (current_dist < target_dist)
    {
        target_dist = current_dist;
        target = index;
    }

    if ((tree[index].p[d] - qp[d])*(tree[index].p[d] - qp[d]) > target_dist || other_index == -1)
    {
        return target;
    }

    other = nn(qp, tree, dim, other_index);
    float other_distance = dist(qp, tree, other);

    if (other_distance > target_dist)
    {
        return target;
    }
    return other;
}

// int nn(float *qp, struct Point *tree, float *dists, int dim, int index)
// {
//     float current_dist = dist(qp, tree, index);
//     dists[index] = current_dist;

//     if (tree[index].left == -1 && tree[index].right == -1)
//     {
//         return index;
//     }

//     int target, other, d = dim % 3,
    
//         target_index = tree[index].right,
//         other_index = tree[index].left;
    
//     dim++;

//     if (tree[index].p[d] > qp[d] || target_index == -1)
//     {
//         int temp = target_index;

//         target_index = other_index;
//         other_index = temp;
//     }

//     target = nn(qp, tree, dists, dim, target_index);
//     float target_dist = dists[target];

//     if (current_dist < target_dist)
//     {
//         target_dist = current_dist;
//         target = index;
//     }

//     if ((tree[index].p[d] - qp[d])*(tree[index].p[d] - qp[d]) > target_dist || other_index == -1)
//     {
//         return target;
//     }

//     other = nn(qp, tree, dists, dim, other_index);
//     float other_distance = dists[other];

//     if (other_distance > target_dist)
//     {
//         return target;
//     }
//     return other;
// }

int nearest(float *qp, struct Point *tree, int lower, int upper, int dim, int n)
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

    if (tree[r].p[d] > qp[d])
    {
        target_lower = lower;
        target_upper = r;
        other_lower = r + 1;
        other_upper = upper;
    }

    target = nearest(qp, tree, target_lower, target_upper, dim, n);

    float target_dist = dist(qp, tree, target),
        current_dist = dist(qp, tree, r);

    if (current_dist < target_dist)
    {
        target_dist = current_dist;
        target = r;
    }

    if ((tree[r].p[d] - qp[d])*(tree[r].p[d] - qp[d]) > target_dist)
    {
        return target;
    }

    other = nearest(qp, tree, other_lower, other_upper, dim, n);

    float other_distance = dist(qp, tree, other);

    if (other_distance > target_dist)
    {
        return target;
    }
    return other;
}

void print_tree(struct Point *tree, int level, int lower, int upper)
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
    printf("(%3.1f, %3.1f, %3.1f) - i:%d, l:%d, r:%d\n", tree[r].p[0], tree[r].p[1], tree[r].p[2], r, tree[r].left, tree[r].right);

    print_tree(tree, 1 + level, lower, r);
    print_tree(tree, 1 + level, r + 1, upper);
}

double wall_time()
{
    struct timeval tmpTime;
    gettimeofday(&tmpTime, NULL);
    return tmpTime.tv_sec + tmpTime.tv_usec/1.0e6;
}

int test_nn(struct Point *tree, int n, float qx, float qy, float qz, float ex, float ey, float ez)
{
    float dists[n];

    float query_point[3];
    query_point[0] = qx, query_point[1] = qy, query_point[2] = qz;
    
    // int best_fit = nn(query_point, tree, dists, 0, midpoint(0, n));
    int best_fit = nn(query_point, tree, 0, midpoint(0, n));

    float actual = tree[best_fit].p[0] + tree[best_fit].p[1] + tree[best_fit].p[2];
    float expected = ex + ey + ez;

    if (actual == expected)
    {
        return 0;
    }
    return 1;
}

int test_nearest(struct Point *tree, int n, float qx, float qy, float qz, float ex, float ey, float ez)
{
    float query_point[3];
    query_point[0] = qx, query_point[1] = qy, query_point[2] = qz;
    
    int best_fit = nearest(query_point, tree, 0, n, 0, n);

    float actual = tree[best_fit].p[0] + tree[best_fit].p[1] + tree[best_fit].p[2];
    float expected = ex + ey + ez;

    if (actual == expected)
    {
        return 0;
    }
    return 1;

    // printf("Closest point to (%3.1f, %3.1f, %3.1f) was (%3.1f, %3.1f, %3.1f) located at %d\n",
    //     query_point[0], query_point[1], query_point[2],
    //     tree[best_fit].p[0], tree[best_fit].p[1], tree[best_fit].p[2],
    //     best_fit);
    // printf("==================\n");
}

void randomPoint(struct Point *x)
{
    x->p[0] = rand() % 1000;
    x->p[1] = rand() % 1000;
    x->p[2] = rand() % 1000;
}

int main(int argc, char *argv[])
{
    int i, j, n = 10, wn = 6, debug = 0;
    struct Point *points, *wiki;

    if (!debug)
    {
        n = 5100000;
        // n = atoi(argv[1]);
    }

    points = malloc(n * sizeof(struct Point));

    srand(time(NULL));
    for (i = 0; i < n; ++i)
    {
        for (j = 0; j < 3; ++j)
        {
            randomPoint(&points[i]);
        }
    }

    if (!debug)
    {
        double time = wall_time();

        build_kd_tree(points, n);
        store_locations(points, 0, n, n);

        double build_time = (wall_time() - time) * 1000;
        // printf("\nBuild duration for %d points: %lf (ms)\n", n, (wall_time() - time) * 1000);

        int test_runs = 10000;
        float **query_data = malloc(test_runs * 3 * sizeof(float));

        for (i = 0; i < test_runs; ++i)
        {
            float *qp = malloc(3 * sizeof(float));
            qp[0] = rand() % 1000;
            qp[1] = rand() % 1000;
            qp[2] = rand() % 1000;
            query_data[i] = qp;
        }

        // printf("\nOld query:\n");

        time = wall_time();
        for (i = 0; i < test_runs; i++) {
            nearest(query_data[i], points, 0, n, 0, n);
        }
        // printf("%lf\n", ((wall_time() - time) * 1000) / test_runs);

        // printf("Average query duration: %lf (ms)\n", ((wall_time() - time) * 1000) / test_runs);
        // printf("Total time for %d queries: %lf (ms)\n", test_runs, ((wall_time() - time) * 1000));

        // printf("\nNew query:\n");

        float *dists = malloc(n * sizeof(float));

        time = wall_time();
        for (i = 0; i < test_runs; i++) {
            // nn(query_data[i], points, dists, 0, midpoint(0, n));
            nn(query_data[i], points, 0, midpoint(0, n));
        }
        printf("%lf\n", ((wall_time() - time) * 1000) / test_runs);

        // printf("Average query duration: %lf (ms)\n", ((wall_time() - time) * 1000) / test_runs);
        // printf("Total time for %d queries: %lf (ms)\n", test_runs, ((wall_time() - time) * 1000));

        free(dists);

        for (i = 0; i < test_runs; ++i)
        {
            free(query_data[i]);
        }
        free(query_data);
    }

    if (debug)
    {
        wiki = malloc(wn * sizeof(struct Point));

        // (2,3), (5,4), (9,6), (4,7), (8,1), (7,2).
        wiki[0].p[0] = 2, wiki[0].p[1] = 3, wiki[0].p[2] = 0;
        wiki[1].p[0] = 5, wiki[1].p[1] = 4, wiki[1].p[2] = 0;
        wiki[2].p[0] = 9, wiki[2].p[1] = 6, wiki[2].p[2] = 0;
        wiki[3].p[0] = 4, wiki[3].p[1] = 7, wiki[3].p[2] = 0;
        wiki[4].p[0] = 8, wiki[4].p[1] = 1, wiki[4].p[2] = 0;
        wiki[5].p[0] = 7, wiki[5].p[1] = 2, wiki[5].p[2] = 0;

        printf("initial list:\n");
        print_tree(points, 0, 0, n);
        printf("==================\n");

        printf("kd tree:\n");
        build_kd_tree(points, n);
        store_locations(points, 0, n, n);
        print_tree(points, 0, 0, n);
        printf("==================\n");

        printf("initial wiki:\n");
        print_tree(wiki, 0, 0, wn);
        printf("==================\n");

        printf("wiki tree:\n");
        build_kd_tree(wiki, wn);
        store_locations(wiki, 0, wn, wn);
        print_tree(wiki, 0, 0, wn);
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

        if (not_passed_test)
        {
            printf("nearest function not working right!\n");
            printf("==================\n");   
        }
        else {
            printf("nearest function still works!\n");
            printf("==================\n");
        }

        not_passed_test = test_nn(wiki, wn, 2, 3, 0, 2, 3, 0)
            + test_nn(wiki, wn, 5, 4, 0, 5, 4, 0)
            + test_nn(wiki, wn, 9, 6, 0, 9, 6, 0)
            + test_nn(wiki, wn, 4, 7, 0, 4, 7, 0)
            + test_nn(wiki, wn, 8, 1, 0, 8, 1, 0)
            + test_nn(wiki, wn, 7, 2, 0, 7, 2, 0)
            + test_nn(wiki, wn, 10, 10, 0, 9, 6, 0)
            + test_nn(wiki, wn, 0, 0, 0, 2, 3, 0)
            + test_nn(wiki, wn, 4, 4, 0, 5, 4, 0)
            + test_nn(wiki, wn, 3, 2, 0, 2, 3, 0)
            + test_nn(wiki, wn, 2, 6, 0, 4, 7, 0)
            + test_nn(wiki, wn, 10, 0, 0, 8, 1, 0)
            + test_nn(wiki, wn, 0, 10, 0, 4, 7, 0);

        if (not_passed_test)
        {
            printf("nn function not working right!\n");
            printf("==================\n");   
        }
        else {
            printf("nn function still works!\n");
            printf("==================\n");
        }

        free(wiki);
    }
    free(points);
    return 0;
}

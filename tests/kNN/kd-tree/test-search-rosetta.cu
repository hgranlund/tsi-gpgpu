#include <knn_gpgpu.h>
#include <float.h>
#include <sys/time.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "test-common.cuh"

#define MAX_DIM 3

struct kd_node_t
{
    float x[MAX_DIM];
    struct kd_node_t *left, *right;
};

void print_tree(struct kd_node_t *root, int level)
{
    if (!root) return;

    int i;

    printf("|");
    for (i = 0; i < level; ++i)
    {
        printf("----");
    }
    printf("(%3.1f, %3.1f, %3.1f)\n", root->x[0], root->x[1], root->x[2]);

    print_tree(root->left, 1 + level);
    print_tree(root->right, 1 + level);
}

inline float dist(struct kd_node_t *a, struct kd_node_t *b, int dim)
{
    float t,
          d = 0;
    while (dim--)
    {
        t = a->x[dim] - b->x[dim];
        d += t * t;
    }
    return d;
}

void swap(struct kd_node_t *x, struct kd_node_t *y)
{
    float tmp[MAX_DIM];
    memcpy(tmp,  x->x, sizeof(tmp));
    memcpy(x->x, y->x, sizeof(tmp));
    memcpy(y->x, tmp,  sizeof(tmp));
}

/* see quickselect method */
struct kd_node_t *find_median(struct kd_node_t *start, struct kd_node_t *end, int idx)
{
    if (end <= start) return NULL;
    if (end == start + 1)
        return start;

    struct kd_node_t *p, *store, *md = start + (end - start) / 2;
    float pivot;
    while (1)
    {
        pivot = md->x[idx];

        swap(md, end - 1);
        for (store = p = start; p < end; p++)
        {
            if (p->x[idx] < pivot)
            {
                if (p != store)
                    swap(p, store);
                store++;
            }
        }
        swap(store, end - 1);

        /* median has duplicate values */
        if (store->x[idx] == md->x[idx])
            return md;

        if (store > md) end = store;
        else        start = store;
    }
}

struct kd_node_t *make_tree(struct kd_node_t *t, int len, int i, int dim)
{
    struct kd_node_t *n;

    if (!len) return 0;

    if ((n = find_median(t, t + len, i)))
    {
        i = (i + 1) % dim;
        n->left  = make_tree(t, n - t, i, dim);
        n->right = make_tree(n + 1, t + len - (n + 1), i, dim);
    }
    return n;
}

/* global variable, so sue me */
int visited;

void nearest(struct kd_node_t *root, struct kd_node_t *nd, int i, int dim,
             struct kd_node_t **best, float *best_dist)
{
    float d, dx, dx2;

    if (!root) return;
    d = dist(root, nd, dim);
    dx = root->x[i] - nd->x[i];
    dx2 = dx * dx;

    visited ++;

    if (!*best || d < *best_dist)
    {
        *best_dist = d;
        *best = root;
    }

    // /* if chance of exact match is high */
    // if (!*best_dist) return;

    // printf("(%3.1f, %3.1f, %3.1f): best_dist = %3.1f, dx2 = %3.1f, dim = %d\n",
    //        root->x[0], root->x[1], root->x[2], *best_dist, dx2, i);

    if (++i >= dim) i = 0;

    nearest(dx > 0 ? root->left : root->right, nd, i, dim, best, best_dist);
    if (dx2 >= *best_dist) return;
    nearest(dx > 0 ? root->right : root->left, nd, i, dim, best, best_dist);
}

#define rand1() (rand() / (float)RAND_MAX)
#define rand_pt(v) { v.x[0] = rand1(); v.x[1] = rand1(); v.x[2] = rand1(); }

double wall_time ()
{
    struct timeval tmpTime;
    gettimeofday(&tmpTime, NULL);
    return tmpTime.tv_sec + tmpTime.tv_usec / 1.0e6;
}

void readPoints(const char *file_path, int n, kd_node_t *points)
{
    FILE *file = fopen(file_path, "rb");
    if (file == NULL)
    {
        fputs ("File error\n", stderr);
        exit (1);
    }
    for (int i = 0; i < n; ++i)
    {
        fread(&points[i].x, sizeof(float), 3, file);
        for (int j = 0; j < 3; ++j)
        {
            points[i].x[j] = round(points[i].x[j] / 100000000.0);
        }
    }

    fclose(file);
}

TEST(search_rosetta, timing)
{
    int i, n;

    struct kd_node_t *root,
            *qp_points,
            *found,
            *million;

    float best_dist;

    srand(time(0));

    for (n = 1000; n <= 10000; n += 1000)
{
        million = (struct kd_node_t *) calloc(n, sizeof(struct kd_node_t));
        qp_points = (struct kd_node_t *) calloc(n, sizeof(struct kd_node_t));
        readPoints("../tests/data/10000_points.data", n, million);
        // readPoints("/home/simenhg/workspace/tsi-gpgpu/tests/data/100_mill_points.data", n, million);

        for (i = 0; i < n; ++i)
        {
            struct kd_node_t point;
            point.x[0] = million[i].x[0];
            point.x[1] = million[i].x[1];
            point.x[2] = million[i].x[2];
            qp_points[i] = point;
        }
        root = make_tree(million, n, 0, 3);

        // print_tree(root, 0);

        int sum = 0,
            test_runs = n;

        double start_time = wall_time();
        for (i = 0; i < test_runs; i++)
        {
            found = 0;
            visited = 0;
            nearest(root, &qp_points[i], 0, 3, &found, &best_dist);
            sum += visited;




            // printf("Looking for (%3.1f, %3.1f, %3.1f), found (%3.1f, %3.1f, %3.1f)\n",
            //        qp_points[i].x[0], qp_points[i].x[1], qp_points[i].x[2],
            //        found->x[0], found->x[1], found->x[2]);

        }
        printf("Time = %lf ms, Size = %d Elements, Awg visited = %3.1f\n", ((wall_time() - start_time) * 1000), n, sum / (float)test_runs);
    };

    free(million);
};


// TEST(search_rosetta, Wikipedia)
// {
//     struct kd_node_t wp[] =
//     {
//         {{2, 3, 0}}, {{5, 4, 0}}, {{9, 6, 0}}, {{4, 7, 0}}, {{8, 1, 0}}, {{7, 2, 0}}
//     };
//     struct kd_node_t *root, *found;
//     float best_dist;

//     root = make_tree(wp, sizeof(wp) / sizeof(wp[1]), 0, 3);


//     for (int i = 0; i < 6; ++i)
//     {
//         visited = 0;
//         found = 0;
//         nearest(root, &wp[i], 0, 3, &found, &best_dist);
//         printf(">> WP tree\nsearching for (%g, %g)\n"
//                "found (%g, %g) dist %g\nseen %d nodes\n\n",
//                wp[i].x[0], wp[i].x[1],
//                found->x[0], found->x[1], sqrt(best_dist), visited);
//     }
// }



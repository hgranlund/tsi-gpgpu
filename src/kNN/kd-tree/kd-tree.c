#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

struct node
{
    double x[3];
    struct node *left, *right;
};

struct node* median(struct node *start, struct node *end, int idx)
{
    if (end <= start)
    {
        return NULL;
    } if (end == start + 1)
    {
        return start;
    }
 
    inline void swap(struct node *x, struct node *y) {
        double tmp[3];
        memcpy(tmp,  x->x, sizeof(tmp));
        memcpy(x->x, y->x, sizeof(tmp));
        memcpy(y->x, tmp,  sizeof(tmp));
    }
 
    struct node *p, *store, *md = start + (end - start) / 2;
    double pivot;
    while (1) {
        pivot = md->x[idx];
 
        swap(md, end - 1);
        for (store = p = start; p < end; p++) {
            if (p->x[idx] < pivot) {
                if (p != store)
                {
                    swap(p, store);
                }
                store++;
            }
        }
        swap(store, end - 1);

        if (store->x[idx] == md->x[idx])
        {
            return md;
        }
 
        if (store > md)
        {
            end = store;
        } else
        {
            start = store;
        }
    }
}
 
struct node* makeKdTree(struct node *t, int len, int i)
{
    struct node *n;
 
    if (!len) return 0;
 
    if ((n = median(t, t + len, i))) {
        i = (i + 1) % 3;
        n->left  = makeKdTree(t, n - t, i);
        n->right = makeKdTree(n + 1, t + len - (n + 1), i);
    }
    return n;
}

double euclidDistance(struct node a, struct node b)
{
    return sqrt(pow((a.x[0] - b.x[0]), 2) + pow((a.x[1] - b.x[1]), 2) + pow((a.x[2] - b.x[2]), 2));
}

void randomPoint(struct node *v)
{
    v->x[0] = rand();
    v->x[1] = rand();
    v->x[2] = rand();
}

void printTree(struct node *root, int height)
{
    if (root)
    {
        printf("Level %d: %lf, %lf, %lf\n", height, root->x[0], root->x[1], root->x[2]);
        printTree(root->left, (height + 1));
        printTree(root->right, (height + 1));
    }
}

double WallTime()
{
    struct timeval tmpTime;
    gettimeofday(&tmpTime, NULL);
    return tmpTime.tv_sec + tmpTime.tv_usec/1.0e6;
}

int main(int argc, char *argv[])
{
    srand(time(NULL));

    int i, N = atoi(argv[1]);//pow(2, 2 + 1) - 1;
    struct node *tree, *root;

    tree = calloc(N, sizeof(struct node));

    for (i = 0; i < N; i++)
    {
        randomPoint(&tree[i]);
    }

    double time = WallTime();
    root = makeKdTree(tree, N, 0);
    
    printf("Build duration for %d points: %lf (ms)\n", N, (WallTime() - time) * 1000);
    // printf("k-d tree:\n");
    // printTree(root, 0);

    free(tree);
    return 0;
}
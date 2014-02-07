#include <stdio.h>
#include <stdlib.h>
#include <time.h>

struct node
{
    double x, y, z;
    struct node *left, *right;
};

void randomPoint(struct node v)
{
    v.x = rand();
    v.y = rand();
    v.z = rand();
}

void printTree(struct node *root, int height)
{
    if (root)
    {
        printf("Level %d: %lf, %lf, %lf\n", height, root->x, root->y, root->z);
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

int main(void)
{
    srand(time(NULL));

    int i, N = 10;
    struct node *root, *left, *right;

    root = calloc(N, sizeof(struct node));

    for (i = 0; i < N; i++)
    {
        randomPoint(root[i]);
    }

    printTree(root, 0);

    free(root);
    return 0;
}
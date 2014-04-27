#include "knn-common.cuh"
#include "math.h"

void populatePoints(struct Point *points, int n)
{
    int i;
    float temp;
    srand(time(NULL));

    for (i = 0; i < n; ++i)
    {
        struct Point t;
        temp = n - i - 1;

        t.p[0] = temp, t.p[1] = temp, t.p[2] = temp;

        points[i] = t;
    }
}

void populatePointSs(struct PointS *points, int n)
{
    int i;
    float temp;
    srand(time(NULL));

    for (i = 0; i < n; ++i)
    {
        struct PointS t;
        temp = n - i - 1;

        t.p[0] = temp, t.p[1] = temp, t.p[2] = temp;

        points[i] = t;
    }
}


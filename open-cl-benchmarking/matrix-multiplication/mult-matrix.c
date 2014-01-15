////////////////////////////////////////////////
// Matrix multiplication of two square matrices
// Benchmark test using naive implementation adapted from:
// http://rosettacode.org/wiki/Matrix_multiplication#C
//
// Adapted by: Teodor A. Elstad (teodor@andeelstad.com) 2014

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

typedef struct { int h, w; double *x;} matrix_t, *matrix;

inline double dot(double *a, double *b, int len, int step)
{
    double r = 0;
    while (len--) {
        r += *a++ * *b;
        b += step;
    }
    return r;
}

matrix mat_new(int h, int w)
{
    matrix r = malloc(sizeof(matrix_t) + sizeof(double) * w * h);
    r->h = h, r->w = w;
    r->x = (double*)(r + 1);
    return r;
}

matrix mat_mul(matrix a, matrix b)
{
    matrix r;
    double *p, *pa;
    int i, j;
    if (a->w != b->h) return 0;

    r = mat_new(a->h, b->w);
    p = r->x;
    for (pa = a->x, i = 0; i < a->h; i++, pa += a->w)
        for (j = 0; j < b->w; j++)
            *p++ = dot(pa, b->x + j, a->w, b->w);
    return r;
}

double WallTime ()
{
  struct timeval tmpTime;
  gettimeofday(&tmpTime,NULL);
  return tmpTime.tv_sec + tmpTime.tv_usec/1.0e6;
}

int main(int argc, char *argv[])
{
    int i,
        x = atoi(argv[1]),
        size = x * x;

    double pi = 3.1415,
           *M;

    M = malloc(size * sizeof(double));

    for (i = 0; i < size; ++i)
    {
        M[i] = pi;
    }

    matrix_t a = { x, x, M };

    double time = WallTime();

    matrix S = mat_mul(&a, &a);

    printf("runtime was %f (ms)\n", (WallTime() - time) * 1000);
    printf("on a matrix of size %d.\n", x);

    free(M);
    free(S);

    return 0;
}

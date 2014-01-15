////////////////////////////////////////////////
// Matrix multiplication of two square matrices
// Benchmark test using the ATLAS implementation of BLAS
//
// Author: Teodor A. Elstad (teodor@andeelstad.com) 2014

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cblas.h>

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
           *M,
           *S;

    M = malloc(size * sizeof(double));
    S = malloc(size * sizeof(double));

    for (i = 0; i < size; ++i)
    {
        M[i] = pi;
        S[i] = 0.0;
    }

    double time = WallTime();

    // BLAS level 3 matrix multiplication
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
        x, x, x, 1.0, M, x, M, x, 0.0, S, x);

    // for test automation
    printf("%f\n", (WallTime() - time) * 1000);

    // for documentation
    // printf("runtime was %f (ms)\n", (WallTime() - time) * 1000);
    // printf("on a matrix of size %d.\n", x);

    free(M);
    free(S);

    return 0;
}
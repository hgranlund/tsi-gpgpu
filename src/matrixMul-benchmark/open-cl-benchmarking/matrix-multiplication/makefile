CC=gcc

mult-matrix: mult-matrix.c
	$(CC) mult-matrix.c -o mult-matrix

blas-mult-matrix: blas-mult-matrix.c
	$(CC) blas-mult-matrix.c -o blas-mult-matrix -lblas -lm

opencl-mult-matrix: opencl-mult-matrix.c
	$(CC) -Wall -g opencl-mult-matrix.c -o opencl-mult-matrix -l OpenCL

all: mult-matrix.c blas-mult-matrix.c opencl-mult-matrix.c
	$(CC) mult-matrix.c -o mult-matrix
	$(CC) blas-mult-matrix.c -o blas-mult-matrix -lblas -lm
	$(CC) -Wall -g opencl-mult-matrix.c -o opencl-mult-matrix -l OpenCL
#include <quick-select.cuh>
#include "test-common.cuh"

#define THREADS_PER_BLOCK 1024U
#define MAX_BLOCK_DIM_SIZE 65535U

TEST(quick_selection, correctness)
{
    struct Point *h_points, *d_points;
    int  *d_steps, *h_steps, n, p, step, dim = 0;
    for (n = 10; n <= 5000; n += 1000)
    {
        p = 2;
        h_points = (struct Point *) malloc(n  * sizeof(Point));
        h_steps = (int *) malloc(p * 2 * sizeof(int));

        h_steps[0] = 0;
        h_steps[1] = n / p;
        h_steps[2] = n / p + 1;
        h_steps[3] = n;
        step = h_steps[1] - h_steps[0];

        checkCudaErrors(cudaMalloc((void **)&d_points, n  * sizeof(Point)));
        checkCudaErrors(cudaMalloc((void **)&d_steps, p * 2  * sizeof(Point)));

        // populatePointSs(h_points, n);
        readPoints("../tests/data/10000_points.data", n, h_points);

        checkCudaErrors(cudaMemcpy(d_points, h_points, n  * sizeof(Point), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_steps, h_steps, p * 2  * sizeof(int), cudaMemcpyHostToDevice));

        quickSelectAndPartition(d_points, d_steps, step , p, dim);

        checkCudaErrors(cudaMemcpy(h_points, d_points, n  * sizeof(Point), cudaMemcpyDeviceToHost));
        ASSERT_TREE_LEVEL_OK(h_points, h_steps, n, p, dim);

        checkCudaErrors(cudaFree(d_points));
        checkCudaErrors(cudaFree(d_steps));
        free(h_steps);
        free(h_points);
        cudaDeviceReset();
    }
}

TEST(quick_selection, correctness_dim)
{
    struct Point *h_points, *d_points;
    int  *d_steps, *h_steps, n, p, step, dim;
    p = 2;
    n = 1024;

    h_points = (struct Point *) malloc(n  * sizeof(Point));
    h_steps = (int *) malloc(p * 2 * sizeof(int));

    h_steps[0] = 0;
    h_steps[1] = n / p;
    h_steps[2] = n / p + 1;
    h_steps[3] = n;
    step = h_steps[1] - h_steps[0];

    checkCudaErrors(cudaMalloc((void **)&d_points, n  * sizeof(Point)));
    checkCudaErrors(cudaMalloc((void **)&d_steps, p * 2  * sizeof(Point)));

    readPoints("../tests/data/10000_points.data", n, h_points);

    checkCudaErrors(cudaMemcpy(d_points, h_points, n  * sizeof(Point), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_steps, h_steps, p * 2  * sizeof(int), cudaMemcpyHostToDevice));

    for (dim = 0; dim < 3; dim++)
    {
        quickSelectAndPartition(d_points, d_steps, step , p, dim);

        checkCudaErrors(cudaMemcpy(h_points, d_points, n  * sizeof(Point), cudaMemcpyDeviceToHost));

        ASSERT_TREE_LEVEL_OK(h_points, h_steps, n, p, dim);

    }

    checkCudaErrors(cudaFree(d_points));
    checkCudaErrors(cudaFree(d_steps));
    free(h_steps);
    free(h_points);
    cudaDeviceReset();
}

TEST(quick_selection, timing)
{
    struct Point *h_points, *d_points;
    int  *d_steps, *h_steps, step, n, p;
    for (n = 2048; n <= 2048; n <<= 1)
    {
        p = 2;
        h_steps = (int *) malloc(p * 2 * sizeof(int));
        h_points = (struct Point *) malloc(n  * sizeof(Point));
        h_steps[0] = 0;
        h_steps[1] = n / p;
        h_steps[2] = n / p + 1;
        h_steps[3] = n;
        step = h_steps[1] - h_steps[0];

        checkCudaErrors(cudaMalloc((void **)&d_points, n  * sizeof(Point)));
        checkCudaErrors(cudaMalloc((void **)&d_steps, p * 2  * sizeof(Point)));

        populatePointSs(h_points, n);
        checkCudaErrors(cudaMemcpy(d_points, h_points, n  * sizeof(Point), cudaMemcpyHostToDevice));

        float elapsed_time = 0;
        cudaEvent_t start, stop;
        cudaStartTiming(start, stop, elapsed_time);

        checkCudaErrors(cudaMemcpy(d_points, h_points, n  * sizeof(Point), cudaMemcpyHostToDevice));
        quickSelectAndPartition(d_points, d_steps, step , p, 0);

        cudaStopTiming(start, stop, elapsed_time);

        int bytes = n * (sizeof(float)) ;
        printCudaTiming(elapsed_time, bytes, n);

        checkCudaErrors(cudaFree(d_points));
        checkCudaErrors(cudaFree(d_steps));
        free(h_steps);
        free(h_points);
        cudaDeviceReset();
    }
}


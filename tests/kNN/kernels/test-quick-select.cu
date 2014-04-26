#include <quick-select.cuh>
#include "test-common.cuh"

#define THREADS_PER_BLOCK 1024U
#define MAX_BLOCK_DIM_SIZE 65535U

TEST(quick_selection, correctness)
{
    struct PointS *h_points, *d_points;
    int  *d_steps, *h_steps, n, p, step, dim = 2;
    for (n = 2; n <= 5000; n += 1000)
    {
        p = 2;
        h_points = (struct PointS *) malloc(n  * sizeof(PointS));
        h_steps = (int *) malloc(p * 2 * sizeof(int));

        h_steps[0] = 0;
        h_steps[1] = n / p;
        h_steps[2] = n / p + 1;
        h_steps[3] = n;
        step = h_steps[1] - h_steps[0];

        checkCudaErrors(cudaMalloc((void **)&d_points, n  * sizeof(PointS)));
        checkCudaErrors(cudaMalloc((void **)&d_steps, p * 2  * sizeof(PointS)));

        populatePointSs(h_points, n);
        checkCudaErrors(cudaMemcpy(d_points, h_points, n  * sizeof(PointS), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_steps, h_steps, p * 2  * sizeof(int), cudaMemcpyHostToDevice));

        quickSelectAndPartition(d_points, d_steps, step , p, dim);

        checkCudaErrors(cudaMemcpy(h_points, d_points, n  * sizeof(PointS), cudaMemcpyDeviceToHost));

        ASSERT_TREE_LEVEL_OK(h_points, h_steps, n, p, dim);

        checkCudaErrors(cudaFree(d_points));
        checkCudaErrors(cudaFree(d_steps));
        free(h_steps);
        free(h_points);
        cudaDeviceReset();
    }
}
TEST(quick_selection, timing)
{
    struct PointS *h_points, *d_points;
    int  *d_steps, *h_steps, step, n, p;
    for (n = 2048; n <= 2048; n <<= 1)
    {
        p = 2;
        h_steps = (int *) malloc(p * 2 * sizeof(int));
        h_points = (struct PointS *) malloc(n  * sizeof(PointS));
        h_steps[0] = 0;
        h_steps[1] = n / p;
        h_steps[2] = n / p + 1;
        h_steps[3] = n;
        step = h_steps[1] - h_steps[0];

        checkCudaErrors(cudaMalloc((void **)&d_points, n  * sizeof(PointS)));
        checkCudaErrors(cudaMalloc((void **)&d_steps, p * 2  * sizeof(PointS)));

        populatePointSs(h_points, n);
        checkCudaErrors(cudaMemcpy(d_points, h_points, n  * sizeof(PointS), cudaMemcpyHostToDevice));

        float elapsed_time = 0;
        cudaEvent_t start, stop;
        cudaStartTiming(start, stop, elapsed_time);

        checkCudaErrors(cudaMemcpy(d_points, h_points, n  * sizeof(PointS), cudaMemcpyHostToDevice));
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


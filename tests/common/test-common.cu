#include "test-common.cuh"


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

void cudaStartTiming(cudaEvent_t &start, cudaEvent_t &stop, float &elapsed_time)
{
    elapsed_time = 0;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventRecord(start, 0));
}

void cudaStopTiming(cudaEvent_t &start, cudaEvent_t &stop, float &elapsed_time)
{
    checkCudaErrors(cudaEventRecord(stop, 0));
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
}

void printCudaTiming(float elapsed_time, float bytes, int n)
{
    double throughput = 1.0e-9 * ((double)bytes) / (elapsed_time * 1e-3);

    printf("Throughput = %.4f GB/s, Time = %.5f ms, Size = %u Elements\n", throughput, elapsed_time, n);
}

void ASSERT_TREE_EQ(struct Point *expected_tree, struct Point *actual_tree, int n)
{
    int i, j;

    for (i = 0; i < n; ++i)
    {
        for (j = 0; j < 3; ++j)
        {
            ASSERT_EQ(expected_tree[i].p[j] , actual_tree[i].p[j]) << "failed with i = " << i << " j = " << j ;
        }
    }
}
// ------ multMatrix_kernel.cl ----------

#define BLOCKSIZE 8

// simple matrix multiplication
__kernel void multMatrixSimple(__global float *mO,
                         __global float *mA,
                         __global float *mB,
                         uint widthA, uint widthB)

{
  int globalIdx = get_global_id(0);
  int globalIdy = get_global_id(1);

  float sum =0;

  for (int i=0; i< widthA; i++)
  {
        float tempA = mA[globalIdy * widthA + i];
        float tempB = mB[i * widthB + globalIdx];
        sum += tempA * tempB;
  }

  mO[globalIdy * widthA + globalIdx] = sum;
}
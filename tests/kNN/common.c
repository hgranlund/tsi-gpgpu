#include <math.h>
#include <time.h>
#include <assert.h>


int debug = 0;

void printArray(float* l, int n){
  if (debug)
  {
    printf("[%3.1f", l[0] );
      for (int i = 1; i < n; ++i)
      {
        printf(", %3.1f", l[i] );
      }
      printf("]\n");
    }
  }

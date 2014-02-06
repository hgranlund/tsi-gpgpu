#include "common-debug.h"

void printFloatArray(float* l, int n){
  int i;
  if (debug)
  {
    printf("[%3.1f", l[0] );
      for (i = 1; i < n; ++i)
      {
        printf(", %3.1f", l[i] );
      }
      printf("]\n");
    }
  }

void printIntArray(int* l, int n){
  int i;
  if (debug)
  {
    printf("[%d", l[0] );
      for (i = 1; i < n; ++i)
      {
        printf(", %d", l[i] );
      }
      printf("]\n");
    }
  }



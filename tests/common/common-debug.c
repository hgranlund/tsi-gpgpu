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

void printArray(int* l, int n){
  if (debug)
  {
    printf("[%d", l[0] );
      for (int i = 1; i < n; ++i)
      {
        printf(", %d", l[i] );
      }
      printf("]\n");
    }
  }

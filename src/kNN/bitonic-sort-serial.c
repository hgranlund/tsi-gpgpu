#include <stdio.h>

#include <math.h>
#include <time.h>

#include "bitonic-sort-serial.h"



void merge_up(float *list, int n)
{
    int step = n / 2, i, j, k;
    float temp;
    while (step > 0)
    {
        for (i = 0; i < n; i += step * 2)
        {
            for (j = i, k = 0; k < step; j++, k++)
            {
                if (list[j] > list[j + step])
                {
                    // swap
                    temp = list[j];
                    list[j] = list[j + step];
                    list[j + step] = temp;
                }
            }
        }
        step /= 2;
    }
}

void merge_down(float *list, int n)
{
    int step = n / 2, i, j, k;
    float temp;
    while (step > 0)
    {
        for (i = 0; i < n; i += step * 2)
        {
            for (j = i, k = 0; k < step; j++, k++)
            {
                if (list[j] < list[j + step])
                {
                    // swap
                    temp = list[j];
                    list[j] = list[j + step];
                    list[j + step] = temp;
                }
            }
        }
        step /= 2;
    }
}

void printArray(float *list, int n)
{
    int i;
    printf("[%.f", list[0]);
    for (i = 1; i < n; i++)
    {
        printf(",%.f", list[i]);
    }
    printf("]\n");
}



void bitonic_sort_serial(float *list, int n)
{
    int s, i;
    printArray(list, n);
    for (s = 2; s <= n; s *= 2)
    {
        for (i = 0; i < n;)
        {
            merge_up((list + i), s);
            merge_down((list + i + s), s);
            i += s * 2;
        }
    }
    // printArray(list,n);
}

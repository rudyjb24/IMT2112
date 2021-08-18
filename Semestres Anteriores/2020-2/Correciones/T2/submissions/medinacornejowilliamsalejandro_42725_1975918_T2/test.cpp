#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <chrono>


float producto_pto(float* vector_1, float* vector_2, int n)
{
  float result = 0;
    #pragma omp parallel for reduction(+ : result)
    for (int i = 0; i < n; i++) 
    {   
        result += (vector_1[i] * vector_2[i]);
    }
  return result;
}

int main(){

    int n = 100;
    float* copy = (float*) calloc(n, sizeof(float));
    for (int i = 0; i < n; i++) {
        copy[i] = 1;
    }
    float resultado = producto_pto(copy, copy, n);

    printf("%f\n", resultado);
    

}
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <chrono>

double dot(double* vec_1, double* vec_2, int n, int p)
{
  double res = 0;
  #pragma omp parallel for num_threads(p), shared(res, vec_1, vec_2)
  for (int i=0; i<n; i++)
  {
    res += vec_1[i]*vec_2[i];
  }
  return res;
}


int main(){

    int n = 100;
    double* copy = (double*) calloc(n, sizeof(double));
    for (int i = 0; i < n; i++) {
        copy[i] = 1;
    }
    double resultado = dot(copy, copy, n, 10);

    printf("%f\n", resultado);
    

}
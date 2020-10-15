#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <chrono>


float par_vec_norm(float* vector, int n, int t) {
    float sum = 0;
    #pragma omp parallel for num_threads(t)
    for (int i = 0; i < n; i++) {
        sum += pow(vector[i], 2);
    }
    return sqrt(sum);
}

float par_inner_product(float* vec1, float* vec2, int n, int t) {
    float sum = 0;
    // Nota: reduction(+:sum) hace que cada thread calcule
    // privadamente una porción del resultado, y luego
    // la acumula en sum.
    #pragma omp parallel for reduction(+:sum) num_threads(t)
    for (int i = 0; i < n; i++) {
        sum += vec1[i] * vec2[i];
    }
    return sum;
}

int main(){

    int n = 100;
    float* copy = (float*) calloc(n, sizeof(float));
    for (int i = 0; i < n; i++) {
        copy[i] = 1;
    }
    float resultado = par_inner_product(copy, copy, n, 50);

    printf("%f\n", resultado);
    

}
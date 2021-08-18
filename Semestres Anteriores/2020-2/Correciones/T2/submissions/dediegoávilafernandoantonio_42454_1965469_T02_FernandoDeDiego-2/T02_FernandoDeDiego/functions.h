#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <chrono>


// Declaramos las funciones.
float** matrix(int n, int range);
float* vector(int n, int range);
float* copy_vector(float* vector, int n);
void free_matrix(float** matrix, int n);
void print_matrix(float** matrix, int n);
void print_vector(float* vector, int n);

// Funciones lineales.
float vec_norm(float* vector, int n);
void mat_vec(float* Ab, float** mat, float* vec, int n);
float inner_product(float* vec1, float* vec2, int n);
void b_iteration(float* Ab, float* b, int n);
float mu_iteration(float* Ab, float* b, int n);

// Funciones paralelizadas.
float par_vec_norm(float* vector, int n, int t);
void par_mat_vec(float* Ab, float** mat, float* vec, int n, int t);
float par_inner_product(float* vec1, float* vec2, int n, int t);
void par_b_iteration(float* Ab, float* b, int n, int t);
float par_mu_iteration(float* Ab, float* b, int n, int t);

// Función para correr el algoritmo.
void power_iteration(float** A, float* b, float* mu, int n, int iter);
void par_power_iteration(float** A, float* b, float* mu, int n, int iter, int t);
void run_algorithm(float** A, float* b, float* mu, int n, int iter, int t);
float return_time(float** A, float* b, float* mu, int n, int iter, int t);

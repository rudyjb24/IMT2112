#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <iostream>
#include <math.h>

int n_threads = 10;

//Fuente: Ayudantía sobre mat_vec
float** matrix_generator(int n, int range)
{
  srand((int)time(0));
  float** matrix = (float**) calloc(n, sizeof(float*));

  for (int i = 0; i < n; i++)
  {
    matrix[i] = (float *) calloc(n, sizeof(float));
  }
  for (int i = 0; i < n; i++)
  {
    for (int j = 0; j < n; j++)
    {   
        int valor = 2*range*((float)rand()/(float)RAND_MAX) - range;
        matrix[i][j] = valor;
        matrix[j][i] = valor;
      
    }
  }
  return matrix;
}

void free_matrix(float **matrix, int n)
{
  for (int i = 0; i < n; i++)
  {
    free(matrix[i]);
  }
  free(matrix);
}

void print_matrix(float** matrix, int n)
{
  printf("\n");
  for (int i = 0; i < n; i++) 
  {
      for (int j = 0; j < n; j++)
      {
         printf("%f ", matrix[i][j]); 
      }
    printf("\n");
  }
}

float* vector_generator(int n, int range)
{
  srand((int)time(0));
  float* vector = (float *)calloc(n, sizeof(float));
  for (int i = 0; i < n; i++) 
  {
    vector[i] = 2*range*((int)rand()/(float)RAND_MAX) - range;
  }
  return vector;
}

void print_vector(float *vector, int n)
{
  printf("\n");
  for (int i = 0; i < n; i++) 
  {
      printf("%f\n", vector[i]);
  }
}

float* mat_vec_par(float **matrix, float *vector, int n)
{
    float *result = (float *)calloc(n, sizeof(float));
    #pragma omp parallel for num_threads(n_threads)
    for (int i = 0; i < n; i++) 
    {
        for (int j = 0; j < n; j++)
        {
            result[i] += matrix[i][j] * vector[j];
        }
    }
  return result;
}

float* normalizar_2(float* vector, int n)
{
    float valor = 0;
    float *result = (float *)calloc(n, sizeof(float));
    #pragma omp parallel for num_threads(n_threads)
    for (int i = 0; i < n; i++) 
    {
            valor += pow(vector[i],2);
    }
    
    #pragma omp parallel for num_threads(n_threads)
    for (int i = 0; i < n; i++) 
    {
            result[i] = vector[i]/sqrt(valor);
        }
    
    return result;
}

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

void power_iteration(float **matrix, float *vector, int n, int n_iteraciones)
{
  float* b_k = vector;
  float* vector_mu  = (float *)calloc(n, sizeof(float));
  float mu_k;

  auto start_p = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < n_iteraciones; i++)
  {
    mu_k = producto_pto(b_k, mat_vec_par(matrix, b_k, n), n)/ producto_pto(b_k, b_k, n);

    b_k = normalizar_2(mat_vec_par(matrix, b_k,n), n);

    vector_mu[i] =  mu_k;
    //printf(" -------- b_k --------");
    //print_vector(b_k, n);
    //printf("Iteración = %i \n", i);
    //printf("mu_k = %f \n", mu_k);
    
  }
  //print_vector(b_k,n);
  //print_vector(vector_mu,n_iteraciones);
  auto end_p = std::chrono::high_resolution_clock::now();
  auto execution_p = std::chrono::duration_cast<std::chrono::nanoseconds>( end_p - start_p );
  printf("Time measured: %.3f seconds.\n", execution_p.count()* 1e-9);
  printf("El valor propio obtenido en la última iteración es mu_k = %f \n", mu_k);
    
  free(vector_mu);
  free(b_k);
}

int main() {
  
  int dimension = 1000; // Dimensión de la matriz y vector
  int rango =  5; // Rango de los números aleatorios
  int n_iteraciones = 1000; //Número máximo de iteraciones

  float** A = matrix_generator(dimension, rango);
  float* x = vector_generator(dimension, rango);
  //print_matrix(A,n);
  //print_vector(x,n);

  power_iteration(A, x, dimension, n_iteraciones);

  free_matrix(A, dimension);
  free(x);
  return 0;

}
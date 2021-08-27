#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <iostream>
#include <cmath>
#include <omp.h>


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
      if (j >= i)
      {
        matrix[i][j] = 2*range*((float)rand()/(float)RAND_MAX) - range;
      }
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


float* vector_generator(int n, int range)
{
  srand((int)time(0));
  float* vector = (float *)calloc(n, sizeof(float));
  for (int i = 0; i < n; i++) 
  {
    vector[i] = 2*range*((float)rand()/(float)RAND_MAX) - range;
  }
  return vector;
}


float* mat_vec_par(float **matrix, float *vector, int n, int n_threads)
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


float dot_product_par(float *vector1, float *vector2, int n, int n_threads)
{
  float result = 0.0;
  int i = 0;
  #pragma omp parallel for num_threads(n_threads) private(i) reduction(+: result)
    for (int i=0; i < n; i++){ 
      result += (vector1[i] * vector2[i]);
    }
  return result;
}

float norm_par(float *vector, int n, int n_threads)
{
  float squared_norm = dot_product_par(vector, vector, n, n_threads);
  float result = sqrt(squared_norm);
  return result;
}


int main() {
  
  int dim = 1000;
  int threads = 16;
  int max_iter = 10;
  float mu;
  float** A = matrix_generator(dim, 10);
  float* b = vector_generator(dim, 10);
  printf("Iniciando el algoritmo con dimension %i, %i threads y %i iteraciones \n", dim, threads, max_iter);
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < max_iter; i++)
  {
    float* num_vector = mat_vec_par(A, b, dim, threads);
    float den = norm_par(num_vector, dim, threads);
    mu = dot_product_par(b, num_vector, dim, threads) / dot_product_par(b, b, dim, threads);

    #pragma omp parallel for num_threads(threads)
    for (int j= 0; j<dim; j++)
    {
      b[j] = num_vector[j] / den; 
    }
    printf("En la iteracion %i el valor propio es %f \n", i, mu);
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto execution = std::chrono::duration_cast<std::chrono::nanoseconds>( end - start ); 
  printf("Algoritmo con dimension %i, %i threads y %i iteraciones terminado\n", dim, threads, max_iter);
  printf("Tiempo de computo: %.3f seconds.\n", execution.count() * 1e-9);
  free_matrix(A, dim);
  free(b);
  return 0;

}
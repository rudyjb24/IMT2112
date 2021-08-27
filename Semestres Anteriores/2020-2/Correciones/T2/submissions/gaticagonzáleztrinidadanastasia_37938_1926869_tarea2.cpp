#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <iostream>
#include <omp.h>
#include <cmath>


//Genera una matriz A ∈ R n×n 

float** matrix_generator(int n, int range)
{
  srand((int)time(0) + 1);
  float** matrix = (float**) calloc(n, sizeof(float*));

  for (int i = 0; i < n; i++)
  {
    matrix[i] = (float *) calloc(n, sizeof(float));
  }
  for (int i = 0; i < n; i++)
  {
    for (int j = 0; j < n; j++)
    {

      {
        matrix[i][j] = 2*range*((float)rand()/(float)RAND_MAX) - range;
      }
    }
  }
  return matrix;
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

void free_matrix(float **matrix, int n)
{
  for (int i = 0; i < n; i++)
  {
    free(matrix[i]);
  }
  free(matrix);
}

// Generar vector inicial 

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
    #pragma omp parallel for num_threads(8)
    for (int i = 0; i < n; i++) 
    {
        for (int j = 0; j < n; j++)
        {
            result[i] += matrix[i][j] * vector[j];
        }
    }
  return result;
}

float vec_vec(float *vector1, float *vector2, int n)
{
  float *result = (float *)calloc(n, sizeof(float));
    #pragma omp parallel for num_threads(8)
    for (int i = 0; i < n; i++) 
    {
      result[i] += vector1[i] * vector2[i];
    }
    float a = 0;
    for (int i = 0; i < n; i++) 
    {
      a += result[i];
    }
  return a;
}


float norma(float* x, int n)
{ 
  float result = 0;
    for (int i = 0; i < n; i++) 
    {
      result += x[i] * x[i];
    }
  return sqrt(result);
}


float* division_paralela(float *vector, float x, int n)
{ 
  //float *result = (float *)calloc(n, sizeof(float));
    #pragma omp parallel for num_threads(8)
    for (int i = 0; i < n; i++) 
    {
      vector[i] = vector[i]/x;
    }
  return vector;
}

int main() {
  auto start = std::chrono::high_resolution_clock::now();
  int n = 1000;

  //Genero matriz A y vector x "aleatorios"
  float** A = matrix_generator(n, 1);
  float* y = vector_generator(n, 1);
  //print_matrix(A, n);
  //print_vector(y, n);


  
  //Creo un loop que calcula el mayor valor propio
  for (int i=0;i<1000;i++)
  {
    y = mat_vec_par(A, y, n);
    //norma
    float no = norma(y, n);
    //división entre y y norma
    y = division_paralela(y, no, n);
    //print_vector(y, n);
  
    
  }
  float* la_der = mat_vec_par(A, y , n);
  float la = vec_vec(y, la_der, n);
  printf("\n %f \n", la);

  
  auto end = std::chrono::high_resolution_clock::now();
  auto execution = std::chrono::duration_cast<std::chrono::nanoseconds>( end - start ); 
  printf("Time measured: %.3f seconds.\n", execution.count() * 1e-9);

  free_matrix(A, n);
  free(y);
  return 0;
}
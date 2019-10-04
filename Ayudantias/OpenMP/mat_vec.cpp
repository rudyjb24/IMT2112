#include <stdio.h>
#include <stdlib.h>
#include <ctime>


float **matrix_generator(int n, int range)
{
  srand((int)time(0));
  float **matrix = (float **) malloc(n * sizeof(float *));
  for (int i = 0; i < n; i++)
  {
    matrix[i] = (float *) malloc(n * sizeof(float));
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

void print_matrix(float **matrix, int n)
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

float *vector_generator(int n, int range)
{
  srand((int)time(0));
  float *vector = (float *)malloc(n * sizeof(float));
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

float *mat_vec(float **matrix, float *vector, int n)
{
    float *result = (float *)calloc(n, sizeof(float));
    for (int i = 0; i < n; i++) 
    {
        for (int j = 0; j < n; j++)
        {
            result[i] += matrix[i][j] * vector[j];
        }
    }
  return result;
}

float *mat_vec_par(float **matrix, float *vector, int n)
{
    float *result = (float *)calloc(n, sizeof(float));
    #pragma omp parallel for
    for (int i = 0; i < n; i++) 
    {
        for (int j = 0; j < n; j++)
        {
            result[i] += matrix[i][j] * vector[j];
        }
    }
  return result;
}

int main() {
  
  int n = 10000;
  clock_t start, end, start2, end2;

  start = time(0);
  float **A = matrix_generator(n, 10);
  float *x = vector_generator(n, 10);
  float *b = mat_vec(A, x, n);
  end = time(0);


  //print_matrix(A, n);
  //print_vector(x, n);
  //print_vector(b, n);

  
  start2 = time(0);
  float *bp = mat_vec_par(A, x, n);
  end2 = time(0);

  printf("\nNos demoramos %2f seg\n", (double)(end-start));
  printf("\nNos demoramos %2f seg\n", (double)(end2-start2));
  free_matrix(A, n);
  free(x);
  return 0;
}
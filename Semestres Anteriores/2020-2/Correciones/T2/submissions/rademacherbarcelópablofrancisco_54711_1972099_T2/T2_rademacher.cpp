#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <chrono>
#include <iostream>
#include <math.h>

double** random_matrix(int n, int max)
{
  //genera matriz simetrica aleatorio de tamaño n
  srand((int)time(0));
  double** mat = (double**)calloc(n, sizeof(double*));
  for (int i=0; i<n; i++)
  {
    mat[i] = (double*)calloc(n, sizeof(double));
  }
  for (int i=0; i<n; i++)
  {
    for (int j=i; j<n; j++)
    {
      if (j>=i)
      {
        mat[i][j] = 2*max*(double)rand()/(double)RAND_MAX-max;
        mat[j][i] = mat[i][j];
      }
    }
  }
  return mat;
}

double* random_vector(int n, int max)
{
  srand((int)time(0)+1);
  double* vec = (double*)calloc(n, sizeof(double));
  for (int i=0; i<n; i++)
  {
    vec[i] = 2*max*(double)rand()/(double)RAND_MAX-max;
  }
  return vec;
}

double* mat_vec(double** mat, double* vec, int n, int p)
{
  double* new_vec = (double*)calloc(n, sizeof(double));
  #pragma omp parallel for num_threads(p), shared(mat, vec, new_vec)
  for (int i=0; i<n; i++)
  {
    for (int j=0; j<n; j++)
    {
      new_vec[i] += mat[i][j]*vec[j];
    }
  }
  return new_vec;
}

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

void escalar_vector(double esc, double* vec, int n, int p)
{
  #pragma omp parallel for num_threads(p)
  for (int i=0; i<n; i++)
  {
    vec[i] = vec[i] * esc;
  }
}

void free_matrix(double** matrix, int n)
{
  for (int i=0; i<n; i++)
  {
    free(matrix[i]);
  }
  free(matrix);
}

int main()
{
  int n = 1000;  //tamaño de la matriz y vector
  int max = 30; //maximo valor que toman los elementos
  int p = 10; //cantidad de procesos
  int max_iteraciones = 1000;



  double** mat = random_matrix(n, max);
  double* vec = random_vector(n, max);

  double* vec_old = (double*)calloc(n, sizeof(double));
  double mu_old;
  double mu;
  double p_punto;
  double norma;




  printf("Test 1:\n");

  auto start_1 = std::chrono::high_resolution_clock::now();
  for (int i=0; i<max_iteraciones || abs(mu_old-mu)<0.0001; i++)
  {
    vec_old = vec;
    vec = mat_vec(mat, vec_old, n, p);
    p_punto = dot(vec, vec, n, p);
    norma = sqrt(p_punto);
    mu_old = mu;
    mu = dot(vec_old, vec, n, p) / p_punto;
    escalar_vector(1/norma, vec, n, p);
    free(vec_old);
  }
  auto end_1 = std::chrono::high_resolution_clock::now();
  auto execution_1 = std::chrono::duration_cast<std::chrono::nanoseconds>( end_1 - start_1 ); 
  printf("El valor propio más grande es %f, el anterior fue %f\n", mu, mu_old);
  printf("Este calculo tomó %.3f segundos.\n", execution_1.count() * 1e-9);

  free(vec);
  free_matrix(mat, n);

  

  return 0;
}

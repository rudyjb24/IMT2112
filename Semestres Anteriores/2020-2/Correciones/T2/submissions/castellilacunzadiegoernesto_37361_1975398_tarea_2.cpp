#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <chrono>
#include <iostream>
#include <omp.h>
#include <cmath>

//Ocuparé como comienzo el archivo "mat_vec" de la ayudantía, pido disculpas :(
//Lo usaré para definir la matriz A, el vector b0, y bueno el mat_vec_par.
//Quizás sería más interesante de una matriz arbitraría, pero (y basandome en el archivo "mat_vec"),
// , ocuparé una matriz triangular superior por la simplicidad en la determinación del valor propio más largo.

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

void print_vector(float *vector, int n)
{
  printf("\n");
  for (int i = 0; i < n; i++)
  {
      printf("%f\n", vector[i]);
  }
}

//"print_eigen" es basicamente recibir un vector de tamaño n+1, printear el vector propio de tamaño n y luego el
//  valor propio.
void print_eigen(float *vector, int n)
{
  printf("\n Vector Propio:");
  printf("\n");
  for (int i = 0; i < n; i++)
  {
      printf("%f\n", vector[i]);
  }
  printf("\nValor propio: %f\n", vector[n]);
}

float* mat_vec_par(float **matrix, float *vector, int n, int p)
{
    float *result = (float *)calloc(n, sizeof(float));
    #pragma omp parallel for num_threads(p)
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            result[i] += matrix[i][j] * vector[j];
        }
    }
  return result;
}

float dot_par(float *vector1, float *vector2, int n, int p)
{
    float *result_a = (float *)calloc(n, sizeof(float));
    float result = 0;
    #pragma omp parallel for num_threads(p)
    for (int i = 0; i < n; i++)
    {
      result_a[i] += vector1[i] * vector2[i];
    }
    for (int j = 0; j < n; j++)
    {
      result += result_a[j];
    }
  return result;
}

float norm_par(float *vector, int n, int p)
{
  float norm_2 = dot_par(vector, vector, n, p);
  float result = sqrt(norm_2);
  return result;
}

//Como plan para retornar tanto el valor como el vector propio aproximados, tengo planeado
// retornar un array de largo n+1 donde los 1,...,n valores corresponden al vector y el ultimo al valor propio :)
float* rayleigh_cuocient(float **matrix, int n, int p, int n_iteraciones)
{
  float *result = (float*) calloc(n, sizeof(float));
  float *b = vector_generator(n, 1000);
  float *Ab = mat_vec_par(matrix, b, n, p);
  float mu = 0;
  float norm_ab = norm_par(Ab, n, p);
  for (int k = 0; k < n_iteraciones; k++)
  {
    for (int i = 0; i < n; i++)
    {
      b[i] =  Ab[i]/norm_ab;
    }
    Ab =  mat_vec_par(matrix, b, n, p);
    norm_ab = norm_par(Ab, n, p);
    mu = dot_par(Ab, b, n, p)/dot_par(b, b, n, p);
  }
  for (int i = 0; i < n; i++)
  {
    result[i] =  b[i];
  }
  result[n] = mu;
  return result;
}

float* eigen_upper(float **matrix, int n)
{
  float *eigen = (float*) calloc(n+1, sizeof(float));
  float index = 0;
  for (int i = 0; i < n; i++)
  {
    if (abs(eigen[n]) < abs(matrix[i][i]))
    {
      eigen[n] = matrix[i][i];
      index = i;
    }
  }
  eigen[(int)index] = 1;
  return eigen;
}

int main()
{
  int n = 5;
  int p = 2;
  int n_iteraciones = 1000;


  float** a = matrix_generator(5, 1000);
  //print_matrix(a, n);
  float* eigen = eigen_upper(a, n);
  print_eigen(eigen, n);

  auto start_p = std::chrono::high_resolution_clock::now();
  float* b_mu = rayleigh_cuocient(a, n, p, n_iteraciones);
  auto end_p = std::chrono::high_resolution_clock::now();
  auto execution_p = std::chrono::duration_cast<std::chrono::nanoseconds>( end_p - start_p );


  print_eigen(b_mu, n);
  printf("Time measured: %.3f seconds.\n", execution_p.count()* 1e-9);
  printf("\n Donde podemos ver que efectivamente converge \n \n");


  int n_1 = 100;
  int p_1 = 2;


  float** a_1 = matrix_generator(n_1, 1000);
  //print_matrix(a, n);
  //float* eigen_1 = eigen_upper(a_1, n_1);
  //print_eigen(eigen, n);

  auto start_p_1 = std::chrono::high_resolution_clock::now();
  float* b_mu_1 = rayleigh_cuocient(a_1, n_1, p_1, n_iteraciones);
  auto end_p_1 = std::chrono::high_resolution_clock::now();
  auto execution_p_1 = std::chrono::duration_cast<std::chrono::nanoseconds>( end_p_1 - start_p_1 );


  //print_eigen(b_mu, n);
  printf("Tiempo medido n = 100, p = 2: %.3f segundos.\n", execution_p_1.count()* 1e-9);


  int n_2 = 1000;
  int p_2 = 2;


  float** a_2 = matrix_generator(n_2, 1000);
  //print_matrix(a, n);
  //float* eigen_1 = eigen_upper(a_1, n_1);
  //print_eigen(eigen, n);

  auto start_p_2 = std::chrono::high_resolution_clock::now();
  float* b_mu_2 = rayleigh_cuocient(a_2, n_2, p_2, n_iteraciones);
  auto end_p_2 = std::chrono::high_resolution_clock::now();
  auto execution_p_2 = std::chrono::duration_cast<std::chrono::nanoseconds>( end_p_2 - start_p_2 );


  //print_eigen(b_mu, n);
  printf("Tiempo medido n = 1000, p = 2: %.3f segundos.\n", execution_p_2.count()* 1e-9);



    int n_3 = 1000;
  int p_3 = 4;


  float** a_3 = matrix_generator(n_3, 1000);
  //print_matrix(a, n);
  //float* eigen_1 = eigen_upper(a_1, n_1);
  //print_eigen(eigen, n);

  auto start_p_3 = std::chrono::high_resolution_clock::now();
  float* b_mu_3 = rayleigh_cuocient(a_3, n_3, p_3, n_iteraciones);
  auto end_p_3 = std::chrono::high_resolution_clock::now();
  auto execution_p_3 = std::chrono::duration_cast<std::chrono::nanoseconds>( end_p_3 - start_p_3 );


  //print_eigen(b_mu, n);
  printf("Tiempo medido n = 1000, p = 4: %.3f segundos.\n", execution_p_3.count()* 1e-9);


    int n_4 = 1000;
  int p_4 = 8;


  float** a_4 = matrix_generator(n_4, 1000);
  //print_matrix(a, n);
  //float* eigen_1 = eigen_upper(a_1, n_1);
  //print_eigen(eigen, n);

  auto start_p_4 = std::chrono::high_resolution_clock::now();
  float* b_mu_4 = rayleigh_cuocient(a_4, n_4, p_4, n_iteraciones);
  auto end_p_4 = std::chrono::high_resolution_clock::now();
  auto execution_p_4 = std::chrono::duration_cast<std::chrono::nanoseconds>( end_p_4 - start_p_4 );


  //print_eigen(b_mu, n);
  printf("Tiempo medido n = 1000, p = 8: %.3f segundos.\n", execution_p_4.count()* 1e-9);



  free_matrix(a, n);
  free(eigen);
  free(b_mu);
  free_matrix(a_1, n_1);
  free_matrix(a_2, n_2);
  free_matrix(a_3, n_3);
  free_matrix(a_4, n_4);
  free(b_mu_1);
  free(b_mu_2);
  free(b_mu_3);
  free(b_mu_4);
  return 0;
}

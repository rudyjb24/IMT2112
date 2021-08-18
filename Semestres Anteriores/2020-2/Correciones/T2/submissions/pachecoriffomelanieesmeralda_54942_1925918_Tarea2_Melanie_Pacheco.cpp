#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <iostream>
#include <cmath> 

// Se ocupo el codigo de ayudantía para generar matriz

float** matrix_generator(int n)
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
    matrix[i][j] = 2*((float)rand()/(float)RAND_MAX) - 1;
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


// Se genera vector con codigo sacado de ayudantia

float* vector_generator(int n)
{
  srand((int)time(0) + 1);
  float* vector = (float *)calloc(n, sizeof(float));
  for (int i = 0; i < n; i++) 
  {
    vector[i] = 2*((float)rand()/(float)RAND_MAX) - 1;
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

// Se crea funcion para multiplicar vectores, tambien sacada de ayudantia

float* mat_vec_par(float **matrix, float *vector, int n, int hilos)
{
    float *result = (float *)calloc(n, sizeof(float));
    #pragma omp parallel for num_threads(hilos)
    for (int i = 0; i < n; i++) 
    {
        for (int j = 0; j < n; j++)
        {
            result[i] += matrix[i][j] * vector[j];
        }
    }
  return result;
}

// Se crea funcion para calcular norma de un vector

float norm(float* x, int n, int hilos)
{
    float * result = (float *)calloc(n, sizeof(float));
    #pragma omp parallel for num_threads(hilos)
    for (int i = 0; i < n; i++) 
    {
        result[i] = x[i] * x[i];
    }

    float norma_cuadrado = 0;

    for (int i = 0; i < n; i++)
    {
        norma_cuadrado += result[i];
    }
    float norma = sqrt(norma_cuadrado);
    return norma;
}

// Se crea una funcion para calcular el producto punto entre dos vectores

float producto_punto(float* x1, float* x2, int n, int hilos)
{
    float *result = (float *)calloc(n, sizeof(float));
    #pragma omp parallel for num_threads(hilos)
    for (int i = 0; i < n; i++) 
    {
        result[i] = x1[i] * x2[i] ;
    }

    float prod_punto = 0;

    for (int i = 0; i < n; i++)
    {
        prod_punto += result[i];
    }
    return prod_punto;
}

// Se crea una funcion para divir un vector por un float

float* division_vec(float* x, float y, int n)
{
  #pragma omp parallel for num_threads(8)
    for (int i = 0; i < n; i++) 
    {
      x[i] = x[i] / y;
    }
  return x;
}

int main() {
  
  // Definimos el tamaño de la matriz y cantidad de iteraciones
  int n = 100;
  int loops = 200;
  int hilos = 16;

  // Se crean el vector, matriz aleatoria y algunos parámetros
  float** A = matrix_generator(n);
  float* x = vector_generator(n);
  float val_prop;
  float val_prop_0 = 0;
  float tolerancia = 0.001;
  bool tolenrancia_cumplida = false;

  // se genera la iteracion y corre el tiempo

  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < loops; i++)
  {
    // Generamos el nuevo vector
    float* Ax = mat_vec_par(A, x, n, hilos);
    //print_vector(Ax,n);

    // calculamos el valor propio

    val_prop = producto_punto(x, Ax, n, hilos) / producto_punto(x, x, n, hilos);
    //printf("valor propio: %f \n ", val_prop);

    // Se calcula si converge viendo si la diferencia del valor propio
    // calculado con el anterior es menor a la tolerancia establecida
    float diferencia = abs(val_prop - val_prop_0);

    if (diferencia < tolerancia)
    {
      tolenrancia_cumplida = true;
    }

    val_prop_0 = val_prop;

    // Normalizamos el vector
    Ax = division_vec(Ax, float(norm(Ax, n, hilos)), n);
    //print_vector(Ax,n);
    x = Ax;
  }

  // Termina el tiempo
  auto end = std::chrono::high_resolution_clock::now();
  auto execution = std::chrono::duration_cast<std::chrono::nanoseconds>( end - start ); 
  printf("Tiempo de demora: %.3f segundos.\n", execution.count() * 1e-9);

  //printf("La matriz es\n")
  //print_matrix(A, n);
  //printf("\nvector propio:");
  //print_vector(x, n);
  printf("Valor propio: %f \n", val_prop);
  printf("Convergencia con tolerancia %f: ", tolerancia);
  printf("%s", tolenrancia_cumplida?"true\n":"false\n");
  //printf("%B\n", tolenrancia_cumplida);
  free_matrix(A, n);
  free(x);

  return 0;
}
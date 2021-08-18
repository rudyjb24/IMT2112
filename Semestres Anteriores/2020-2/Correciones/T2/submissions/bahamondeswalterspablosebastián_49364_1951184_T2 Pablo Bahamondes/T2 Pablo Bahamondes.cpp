#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <iostream>
#include <math.h> 


// Holaaa copie ty generador de matriz y le agregué una linea para que también fuera simétrica
float** matriz(int n, int range, int semilla, int hilos) 
{
  srand(semilla); //(int)time(0)
  float** matrix = (float**) calloc(n, sizeof(float*));
  #pragma omp parallel for num_threads(hilos)
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
      else; //Estructura simétrica en la matriz
      {
       matrix[i][j] += matrix[j][i]; 
      }
      
    }
  }
  return matrix;
}

float* vector_inicial(int n, int range)
{
  float* vector = (float *)calloc(n, sizeof(float));
  vector[0] = 1;
  return vector;
}

void free_matriz(float **matrix, int n)
{
  for (int i = 0; i < n; i++)
  {
    free(matrix[i]);
  }
  free(matrix);
}

// Prints
void print_matriz(float** matrix, int n)
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

// Funciones 
float* mat_vec_par(float **matrix, float *vector, int n,int hilos)
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
// Funciones propias
float norma(float *vector, int n, int hilos)
{
  float* resultado = (float *) calloc(n, sizeof(float));

  // Producto punto en paralelo
  #pragma omp parallel for num_threads(hilos)
  for (int i=0; i<n; i++)
  {
    resultado[i] += vector[i]*vector[i];
  }

  // Suma de en serie
  for (int  i=1; i<n; i ++)
  {
    resultado[0] += resultado[i];
  }
  resultado[0] = sqrt(resultado[0]);
  return resultado[0];
}

void normalizar(float* vector, float norma, int n,int hilos){
  #pragma omp parallel for num_threads(hilos)
  for (int i = 0; i<n; i++)
  {
    vector[i] /= norma;
  }
}

float dot_product(float *vector1, float *vector2, int n, int hilos)
{
    float *resultado = (float *)calloc(n, sizeof(float));

    #pragma omp parallel for num_threads(hilos)
    for (int i = 0; i < n; i++) 
    {
        resultado[i] += vector1[i] * vector2[i];
    }

    for (int i = 1; i < n; i++) 
    {
        resultado[0] += resultado[i];
    }
  return resultado[0];
}


int main() {

  int n = 10000;                     // Tamaño inicial matriz
  int semilla = 33;                  // Semilla para generar
  int iteraciones = 100;             // Cantidad de iteraciones (x_1,...,x_iteraciones)
  int crecimiento_matriz = 1;        // Prueba con matrices de distinto tamaño (para rendimiento), 10,100,1000 (va en múltiplos de 10)
  int rango = 10;                    // Rango de valores que puede tomar la matriz (-100,100)
  int hilos = 10; 
  
  bool valor_propio = false;          // Muestra (y calcula) valores propios de cada iteración
  bool vector_propio = false;         // Muestra (y calcula) vectores propios de cada iteración
  bool mostrar_tiempos = true;        // Muestra el tiempo para cada tamaño de matriz
  float* tiempos = (float *)calloc(crecimiento_matriz, sizeof(float)); 

  printf("Número de hilos: %i\n", hilos); 
  for (int j = 0; j < crecimiento_matriz; j++) 
  {
  printf("\nTamaño Matriz: %i\n",n);
  float** A = matriz(n, rango,semilla,hilos);
  float norm;

  auto inicio = std::chrono::high_resolution_clock::now();

  float* x = vector_inicial(n, 1);
  float* y = vector_inicial(n, 1);
  for (int i = 0; i<= iteraciones; i++)
    {
      y = x;                        // y es nuestro "x"
      x = mat_vec_par(A,x,n,hilos); // "x+1" (sin normalizar)

      if (valor_propio or vector_propio or i == iteraciones)
      {
        printf("\nIteracion %i:\n",i);

        if (valor_propio or i== iteraciones)
        {
          float vpropio = dot_product(x,y,n,hilos);
          printf("Valor propio: %f\n",vpropio);
        }

        if (vector_propio)
        {
          printf("Vector propio:\n");
          print_vector(y,n);
        }

      }

      norm = norma(x,n,hilos);
      normalizar(x,norm,n,hilos);

    }
  auto fin = std::chrono::high_resolution_clock::now();
  auto duracion = std::chrono::duration_cast<std::chrono::nanoseconds>( fin - inicio );

  tiempos[j] =  duracion.count() * 1e-9; 
  if (mostrar_tiempos)
  {
  printf("Tiempo de ejecución: %.8f segundos.\n",tiempos[j]);
  }

  float vpropio = dot_product(x,y,n,hilos);

  

  free_matriz(A,n);
  free(x);
  free(y);

  n = n*10;
  }

  free(tiempos);
  return 0;
}

#include <stdio.h>
#include <stdlib.h> 
#include <ctime>
#include <omp.h> // NO OLVIDAR: compilar con -fopenmp
#include <chrono>
#include <iostream>
#include <math.h>

#define thrds 5
#define semilla 24 //(int)time(0)

// (*)= cortesia de Rudy <3
// README flaite: abajo, en el main, se cambia el tamaño de matriz y el numero de iteraciones.
// Aqui,linea9, se puede definir el nuemro de hilos, como solo se subía un cpp no supe donde hacer el analisis.
// converge, probado con numpy



////// EXTRAS, debido a que no existe un num(py)c++////////

//funcion vector*vector (vector -> real)
float vec_vec(float* v1, float* v2, int DIM) 
{
    float result = 0;
    #pragma omp parallel for num_threads(thrds) 
    for (int i = 0; i < DIM; i++)
    {
        result += v1[i] * v2[i];
    }

  return result;
}


//funcion norma  (vector -> real)
float norma_2(float* vec, int DIM){
    float res = vec_vec(vec,vec, DIM);
    return (float)sqrt(res);
}

//funcion divicion elementwise (vector, escalar -> vector)
float* div_vec(float* vec, float dividendo, int DIM){
    float* result = (float *)calloc(DIM, sizeof(float));
    #pragma omp parallel for num_threads(thrds) 
    for (int i = 0; i < DIM; i++)
    {
        result[i] = (float)vec[i]/dividendo;
    }
    return result;
}

//funcion matriz*vector  (matriz, vector ->vector)
float* mat_vec(float** matrix, float* vector, int DIM) //, int thrds)
{
    float* result = (float *)calloc(DIM, sizeof(float));

    #pragma omp parallel for num_threads(thrds) 
    for (int i = 0; i < DIM; i++)  {
        for (int j = 0; j < DIM; j++)  {
            result[i] += matrix[i][j] * vector[j];
        }
    }

  return result; // retorna un array, no olvides liberar
}
// funcion generadora de matrices (*)

//////////////FUNCIONES/////////////////7
float** matrix_generator(int DIM, int range){
  srand(semilla);
  float** matrix = (float**) calloc(DIM, sizeof(float*));
  
  // creamos las columnas para cada fila
  for (int i = 0; i < DIM; i++)  {
    matrix[i] = (float *) calloc(DIM, sizeof(float));
  }
  // y rellenamos
  for (int i = 0; i < DIM; i++)  {
    for (int j = 0; j < DIM; j++) {
    matrix[i][j] = 2*range*((float)rand()/(float)RAND_MAX) - range;  
    }
  }
  return matrix;
}

//funcion liberadora de arreglos de matriz (*)
void free_matrix(float** matrix, int DIM){
  for (int i = 0; i < DIM; i++)
  {
    free(matrix[i]);
  }
  free(matrix);
}

// imprimir matriz (*)
void print_matrix(float** matrix, int DIM)
{
  printf("\n");
  for (int i = 0; i < DIM; i++) 
  {
      for (int j = 0; j < DIM; j++)
      {
         printf("%f ", matrix[i][j]); 
      }
    printf("\n");
  }
}

// funcion generadora de vectores (*)
float* vector_generator(int DIM, int range){
  srand(semilla);
  float* vector = (float *)calloc(DIM, sizeof(float));
  for (int i = 0; i < DIM; i++) 
  {
    vector[i] = 2*range*((float)rand()/(float)RAND_MAX) - range;
  }
  return vector;
}

// imprimir vector (*)
void print_vector(float* vector, int DIM)
{
  printf("\n");
  for (int i = 0; i < DIM; i++) 
  {
      printf("%f\n", vector[i]);
  }
}

// funcion que ejecuta algoritmo de potencias -> vector propio
float* power_iteration_vecprop(float** A, float* b_0, int N_ITER, int DIM){
    float* b_k = (float*)calloc(DIM, sizeof(float)); 
    float* b_k1 = mat_vec(A,b_0,DIM);
    b_k1 = div_vec(b_k1, norma_2(b_k1,DIM),DIM);

    for (int c = 0; c < N_ITER; c++) {
        b_k = div_vec(b_k1,1,DIM); //sort of copy
        b_k1 = mat_vec(A,b_k,DIM);
        b_k1 = div_vec(b_k1, norma_2(b_k1,DIM),DIM);
    }
    free(b_k);
    return b_k1;
}

// funcion que, dado vecprop retorna el valprop
float power_iteration_valprop(float** A, float* valprop, int N_ITER, int DIM){
    return vec_vec(valprop, mat_vec(A, valprop, DIM), DIM)/norma_2(valprop,DIM);
}


////////////////////////////////////////////////
int main(){
    // Definimos parametros
    int DIM = 100;
    int N_ITER = 100;

    // inicializar el valor y vector propio buscado:
    float* v_max = (float*)calloc(10, sizeof(float));
    float mu_max;

    // generamos la matriz y vector
    float** A = matrix_generator(DIM, 1);
    float* b = vector_generator(DIM, 1);
    if (DIM <= 5){
        printf("\nA:");
        print_matrix(A, DIM);
        printf("\nb:");
        print_vector(b,DIM);
    }

    // ejecutamos la funcion
    auto start = std::chrono::high_resolution_clock::now();
    v_max = power_iteration_vecprop(A,b,N_ITER,DIM);
    mu_max = power_iteration_valprop(A,v_max,N_ITER,DIM);
    auto end = std::chrono::high_resolution_clock::now();
    auto execution = std::chrono::duration_cast<std::chrono::nanoseconds>( end - start ); 


    printf("\nvector propio:");
    print_vector(v_max, DIM);

    printf("\nvalor propio: %f", mu_max);
    
    printf("\ndimension: %i",DIM);
    printf("\nn_iter: %i", N_ITER);
    printf("\nn hilos: %i", thrds);
    printf("\nTime measured: %.3f seconds.\n", execution.count() * 1e-9);


    // liberamos el espacio
    free_matrix(A, DIM);
    free(b);
    free(v_max);

    return 0;
}




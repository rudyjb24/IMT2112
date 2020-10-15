#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <iostream>
#include <omp.h>
#include <cmath>


//funcion que genera una matriz simetrica (implica diagonalizable)
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
    //esta linea define la diagonal
    matrix[i][i] = 2*range*((float)rand()/(float)RAND_MAX) - range;
    for (int j = 0; j < i; j++)
    {
      //aca nos aseguramos de la simetria fuera de la diagonal
      float num = 2*range*((float)rand()/(float)RAND_MAX) - range;
      matrix[i][j] = num;
      matrix[j][i] = num;
    }
  }
  return matrix;
}

//libera la memoria de la matriz
void free_matrix(float **matrix, int n)
{
  for (int i = 0; i < n; i++)
  {
    free(matrix[i]);
  }
  free(matrix);
}

//genera un vector con semilla aleatoria
float* vector_generator(int n, int range)
{
  srand(1000*(int)time(0));
  float* vector = (float *)calloc(n, sizeof(float));
  for (int i = 0; i < n; i++)
  {
    vector[i] = 2*range*((float)rand()/(float)RAND_MAX) - range;
  }
  return vector;
}
//matvec paralelo de la ayudantia
float* mat_vec_par(float **matrix, float *vector, int n, int threads)
{
    float *result = (float *)calloc(n, sizeof(float));
    #pragma omp parallel for num_threads(threads)
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            result[i] += matrix[i][j] * vector[j];
        }
    }
    return result;
}
//producto punto paralelo
float prod_punto(float *vector1, float *vector2, int n, int threads)
{
    float result = 0;
    //array que nos ayudara a asegurar thread safe
    float* array = (float *) calloc(threads, sizeof(float));
    //paralelizamos
    #pragma omp parallel for num_threads(threads)
    for (int i = 0; i < n; i++)
    {
      int id = omp_get_thread_num();
      //cada thread solo puede usar su casilla asignada en el array (safe)
      array[id] += vector1[i]*vector2[i];
    }
    //recuperamos la informacion desde el array donde guardamos los resultados
    for (int i=0;i<threads;i++){
      result += array[i];
      }
    free(array);
    return result;
}

float* reescalate_vector(float* vector, int n, float param)
{
  for (int j = 0; j < n; j++)
  {
    vector[j] *= param;
  }
  return vector;
}

float u_k(float** matrix, float* vector, int n, int threads){
  //calculamos Ab_k
  float* m_v = mat_vec_par(matrix,vector,n,threads);
  //luego calculamos b_k^t(Ab_k)/||b_k^tb_k||
  float u = prod_punto(vector, m_v, n,threads)/prod_punto(vector, vector,n,threads);
  //liberamos el vector generado por matvec
  free(m_v);
  return(u);
}

float* b_k(float** matrix, float* vector, int n, int threads){
  float* b;
  b = mat_vec_par(matrix, vector,n, threads);
  b = reescalate_vector(b, n, 1.0/pow(prod_punto(b,b,n,threads),.5));
  free(vector);
  return b;
}

void metodo_potencia(float** A, float* b_i, int n, int threads){
  //aqui comenzaran los calculos, inicializamos el timer
  auto start = std::chrono::high_resolution_clock::now();
  //
  //calculamos el valor inicial u_0, que aproxima al valor propio
  float u_i = u_k(A,b_i,n,threads);
  //calculamos b_1 a mano para no liberar el vector inicial y pasarlo a los
  //otros experimentos
  float* b_f = mat_vec_par(A, b_i,n, threads);
  b_f = reescalate_vector(b_f, n, 1.0/pow(prod_punto(b_f,b_f,n,threads),.5));
  //calculamos u_1
  float u_f = u_k(A,b_f,n,threads);
  //comenzamos el algoritmo, para en mil iteraciones o error menor a 0.0001
  int cuenta = 0;
  while (cuenta < 1000 && abs(u_f - u_i) > pow(10,-4)){
  u_i = u_f;
  b_f = b_k(A,b_f,n,threads);
  u_f = u_k(A,b_f,n,threads);
  cuenta += 1;
  }
  //Aquí termina el algoritmo, paramos el timer:
  auto end = std::chrono::high_resolution_clock::now();
  auto execution = std::chrono::duration_cast<std::chrono::nanoseconds>( end - start );
  printf("El algoritmo se ejecuto en dimension %i\n", n);
  printf("La cantidad de threads utilizados fue %i\n", threads);
  printf("Tiempo empleado: %.3f segundos\n", execution.count() * 1e-9);
  printf("Cantidad de iteraciones hechas: %i\n", cuenta);
  printf("Error de convergencia alcanzado: %f\n", abs(u_i-u_f));
  printf("El valor absoluto del valor propio mas grande es aproximadamente: %f\n", abs(u_f));
  printf("\n\n\n");
  //este vector b_f no se alcanza a liberar despues de salir del while,
  //porque no pasa por la funcion b_k
  free(b_f);
}

int main() {
  //definimos la dimension
  int n = 100;
  //generamos la matriz simetrica
  float** A = matrix_generator(n, 10);
  //generamos un vector aleatorio inicial
  float* b = vector_generator(n, 10);
  //corremos UN METODO DE POTENCIA
  metodo_potencia(A,b,n,10);

  //DESCOMENTAR ESTO PARA PROBAR DISTINTOS CANTIDADES DE HILOS:
  //corremos el metodo variando la cantidad de hilos entre 1 y 8
  //for (int i = 1; i <8; i+=1){
  //metodo_potencia(A,b,n,i);
  //}

  //DESCOMENTAR ESTO PARA VER QUE EL OVERHEAD EMPEORA EL TIEMPO DE EJECUCION
  //corremos con 50 hilos para ver que el overhead es mucho
  //metodo_potencia(A,b,n,50);

  //liberamos los unicos recursos que no liberamos antes
  free(b);
  free_matrix(A, n);
  return 0;
}

// Compile the code: mpic++ helloMPI.cpp
// Run the executable: mpirun -np 2 ./a.out

#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <iostream>
# include <omp.h>
#include <mpi.h>
#include <cmath> 
using namespace std;

// denifimos la funcion alpha
float alpha_function(float x, float y){
    float alpha = x * (x-1) * y * (y-1) + 1;
    return alpha;
}

float f_function(float x, float y){
  float f = y+x;
  return f;
}

float** f_vector(int filas, int filas_index, int Nx)
{
  float** f_vec= (float**) calloc(Nx, sizeof(float*));
  for (int i = 0; i < Nx; i++)
  {
    f_vec[i] = (float*) calloc(filas, sizeof(float));
  }
  // se crearan los las constantes de la funcion f de los nodos
  for (int i = 0; i < Nx; i++)
  {
      for (int j = 0; j < filas; j++)
      {
            f_vec[i][j] = f_function(i,j + filas_index);
      }
    }
return f_vec;
}

void print_matrix(float** matrix, int filas, int columnas)
{
  printf("\n");
  for (int i = 0; i < filas; i++) 
  {
      for (int j = 0; j < columnas; j++)
      {
         printf("%f ", matrix[i][j]); 
      }
    printf("\n");
  }
}

// malla con las constante del norte de los nodos
float** stencil_const_norte(int filas, int filas_index, int Nx, int Ny,
float hx, float hy)
{
    float** const_N= (float**) calloc(Nx, sizeof(float*));
    for (int i = 0; i < Nx; i++)
  {
    const_N[i] = (float*) calloc(filas, sizeof(float));
  }
// se crearan los las constante norte de los nodos
  for (int i = 0; i < Nx; i++){
      for (int j = 0; j < filas; j++){
          if ((j + filas_index) != (Ny-1))
          {
            const_N[i][j] = - alpha_function(i * hx, (j + 0.5 + filas_index) *hy)/ (hy*hy);
          }
      }
    }
return const_N;
}

// malla con las constante del este de los nodos
float** stencil_const_este(int filas, int filas_index, int Nx, int Ny,
float hx, float hy)
{
    float** const_E= (float**) calloc(Nx, sizeof(float*));
    for (int i = 0; i < Nx; i++)
  {
    const_E[i] = (float*) calloc(filas, sizeof(float));
  }
// se crearan los las constante este de los nodos
  for (int i = 0; i < Nx; i++){
      for (int j = 0; j < filas; j++){
          if (i != (Nx-1))
          {
            const_E[i][j] = - alpha_function( (i + 0.5) * hx,
            (j + filas_index) * hy) / (hx*hx);
          }
      }
    }
return const_E;
}

// malla con las constante del sur de los nodos
float** stencil_const_sur(int filas, int filas_index, int Nx, int Ny,
float hx, float hy)
{
    float** const_S= (float**) calloc(Nx, sizeof(float*));
    for (int i = 0; i < Nx; i++)
  {
    const_S[i] = (float*) calloc(filas, sizeof(float));
  }
// se crearan los las constante sur de los nodos
  for (int i = 0; i < Nx; i++){
      for (int j = 0; j < filas; j++){
          if ((j + filas_index) != 0)
          {
            const_S[i][j] = - alpha_function(i * hx, (j - 0.5 + filas_index) *hy)/ (hy*hy);
          }
      }
    }
return const_S;
}

// malla con las constante del oeste de los nodos
float** stencil_const_oeste(int filas, int filas_index, int Nx, int Ny,
float hx, float hy)
{
    float** const_O= (float**) calloc(Nx, sizeof(float*));
    for (int i = 0; i < Nx; i++)
  {
    const_O[i] = (float*) calloc(filas, sizeof(float));
  }
// se crearan los las constante oeste de los nodos
  for (int i = 0; i < Nx; i++){
      for (int j = 0; j < filas; j++){
          if (i != 0)
          {
            const_O[i][j] = -(alpha_function((i - 0.5) * hx, (j + filas_index) * hy) / (hx*hx));
          }
      }
    }
return const_O;
}

// stencil con las constante del centro de los nodos
float** stencil_const_centro(int filas, int filas_index, int Nx, int Ny,
float hx, float hy)
{
    float** const_C= (float**) calloc(Nx, sizeof(float*));
    for (int i = 0; i < Nx; i++)
  {
    const_C[i] = (float*) calloc(filas, sizeof(float));
  }
// se crearan los las constante centro de los nodos
  for (int i = 0; i < Nx; i++){
      for (int j = 0; j < filas; j++)
      {
            const_C[i][j] = ((alpha_function((i - 0.5)*hx, (j+filas_index)*hy) + 
        alpha_function((i + 0.5)*hx, (j+filas_index) * hy)) / (hx*hx)) + 
        ((alpha_function(i*hx, (j+filas_index-0.5) * hy) +
         alpha_function(i*hx, (j+0.5+filas_index)*hy))/ (hy*hy)) + 1;
      }
    }
return const_C;
}

void free_matrix(float **matrix, int filas)
{
  for (int i = 0; i < 1; i++)
  {
    free(matrix[i]);
  }
  free(matrix);
}

void print_vector(float *vector, int n)
{
  printf("\n");
  for (int i = 0; i < n; i++) 
  {
      printf("%f\n", vector[i]);
  }
}

// funcion colectiva para obtener elemtos de procesador anterior
float* elementos_p_anterior(int rank, int size, float** matriz, int Nx, int filas)
{
  float* nodos_enviar = (float*) calloc(Nx, sizeof(float));
  float* nodos_recibir = (float*) calloc(Nx, sizeof(float));
  // Enviar a impares y estos reciben
  if ((rank%2 == 0) && (rank+1 < size))
  {
    for (int i = 0; i < Nx; i++)
    {
      nodos_enviar[i] = matriz[i][filas-1];
    }
    int err = MPI_Send(nodos_enviar, Nx, MPI_FLOAT, rank+1, rank, MPI_COMM_WORLD);
  }
  if ((rank%2 == 1) && (rank >0))
  {
    int err = MPI_Recv(nodos_recibir, Nx, MPI_FLOAT, rank-1, rank-1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
  // Enviar a pares, estos reciben
  if ((rank%2 == 1) && (rank+1 < size))
  {
    //float* nodos_enviar = (float*) calloc(Nx, sizeof(float));
    for (int i = 0; i < Nx; i++)
    {
      nodos_enviar[i] = matriz[i][filas-1];
    }
    int err = MPI_Send(nodos_enviar, Nx, MPI_FLOAT, rank+1, rank, MPI_COMM_WORLD);
  }
  if ((rank%2 == 0) && (rank >0))
  {
    int err = MPI_Recv(nodos_recibir, Nx, MPI_FLOAT, rank-1, rank-1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
  return nodos_recibir;
}

// funcion colectiva para obtener elemtos de procesador posterior
float* elementos_p_posterior(int rank, int size, float** matriz, int Nx, int filas)
{
  float* nodos_enviar = (float*) calloc(Nx, sizeof(float));
  float* nodos_recibir = (float*) calloc(Nx, sizeof(float));
  // Enviar a impares y estos reciben
  if ((rank%2 == 0) && (rank != 0))
  {
    for (int i = 0; i < Nx; i++)
    {
      nodos_enviar[i] = matriz[i][0];
    }
    int err = MPI_Send(nodos_enviar, Nx, MPI_FLOAT, rank-1, rank, MPI_COMM_WORLD);
  }
  if ((rank%2 == 1) && (rank != size-1))
  {
    int err = MPI_Recv(nodos_recibir, Nx, MPI_FLOAT, rank+1, rank+1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
  // Enviar a impares, estos reciben
  if ((rank%2 == 1) && (rank != 0))
  {
    for (int i = 0; i < Nx; i++)
    {
      nodos_enviar[i] = matriz[i][0];
    }
    int err = MPI_Send(nodos_enviar, Nx, MPI_FLOAT, rank-1, rank, MPI_COMM_WORLD);
  }
  if ((rank%2 == 0) && (rank != size -1))
  {
    int err = MPI_Recv(nodos_recibir, Nx, MPI_FLOAT, rank+1, rank+1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
  return nodos_recibir;
}

//suma en paralelo
float** suma_vec(float** mat1, float** mat2, int signo, int Nx, int filas)
{
  float** sum = (float**) calloc(Nx, sizeof(float*));
    for (int i = 0; i < Nx; i++)
  {
    sum[i] = (float*) calloc(filas, sizeof(float));
  }
  // sumaran los valores de cada vector
  for (int i = 0; i < Nx; i++){
      for (int j = 0; j < filas; j++)
      {
            sum[i][j] = mat1[i][j] + mat2[i][j] * signo;
      }
    }
  return sum;
}

//producto punto
float producto_punto(int rank, int size, float** mat1, float** mat2, int Nx, int filas)
{
  float** multi = (float**) calloc(Nx, sizeof(float*));
    for (int i = 0; i < Nx; i++){
    multi[i] = (float*) calloc(filas, sizeof(float));
  }
  // multiplicaran los valores de cada vector
  for (int i = 0; i < Nx; i++){
      for (int j = 0; j < filas; j++){
            multi[i][j] = mat1[i][j] * mat2[i][j];
      }}
  float total = 0;
  //print_matrix(sum_vector, 1, 4);
  for (int i = 0; i < Nx; i++){
      for (int j = 0; j < filas; j++)
      {
        total += multi[i][j];
      }
    }
  
  if (rank == 0)
    {
      float buffer ;
      for (int p=1; p<size; p++)
      {
        int err = MPI_Recv(&buffer, 1, MPI_FLOAT, p, p, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        total += buffer;
      }
      for (int p=1; p<size; p++)
      {
        int err = MPI_Send(&total, 1, MPI_FLOAT, p, rank, MPI_COMM_WORLD);
      }
    }
  else 
  {
      int err = MPI_Send(&total, 1, MPI_FLOAT, 0, rank, MPI_COMM_WORLD);
      err = MPI_Recv(&total, 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
  return total;
}

// escalar (constante) por vector
float** esc_vec(float** mat, float c, int Nx, int filas)
{
  float** mat_es = (float**) calloc(Nx, sizeof(float*));
    for (int i = 0; i < Nx; i++){
    mat_es[i] = (float*) calloc(filas, sizeof(float));
    }
  for (int i = 0; i < Nx; i++){
      for (int j = 0; j < filas; j++)
      {
        mat_es[i][j] = mat[i][j] * c;
      }}
  return mat_es;
}

// Mat vec stencil
float** function_mat_vec(int rank, int size, float** N, float** O, float** S, float** E,
               float** C, float** vec, int Nx, int filas)
{
  float** mat_vec = (float**) calloc(Nx, sizeof(float*));
    for (int i = 0; i < Nx; i++){
     mat_vec[i] = (float*) calloc(filas, sizeof(float));
  }
  float* p_anterior = elementos_p_anterior(rank, size, vec, Nx, filas);
  float* p_posterior = elementos_p_posterior(rank, size, vec, Nx, filas);
  
  for(int i = 0; i<Nx; i++){
      for(int j = 0; j<filas; j++){
        mat_vec[i][j] += C[i][j]*vec[i][j];
        if (i!=0){
          mat_vec[i][j] += O[i][j]*vec[i-1][j];
        }
        if (i!=Nx-1){
          mat_vec[i][j] += E[i][j]*vec[i+1][j];
        }
        // ojo que si estamos en el ultimo procesador no importa ya que tomara
        // un vector de 0
        if (j==filas-1){ 
          mat_vec[i][j] += N[i][j]*p_posterior[i];
        }// ojo que si estamos en el ultimo procesador no importa ya que tomara
        // un vector de 0
        else{
          mat_vec[i][j] += N[i][j]*vec[i][j+1];
        }
        if (j==0){ 
          mat_vec[i][j] += S[i][j]*p_anterior[i];
        }// ojo que si estamos en el primer procesador no importa ya que tomara
        // un vector de 0
        else{
          mat_vec[i][j] += S[i][j]*vec[i][j-1];
        }
      }
    }
  return mat_vec;
}

int main()
{
  MPI_Init(NULL,NULL);
	int world_size, world_rank;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int firstIndex, localfilas, Nx, Ny, err, iteraciones;
    float hx, hy;
    
    Nx = 100;
    Ny = 100;
    hx = 1 /float(Nx);
    hy = 1 /float(Ny);
    iteraciones = 200;
    localfilas = Ny / world_size; //indica las filas que tendra cada bloque
    firstIndex = world_rank*localfilas;

    if (world_rank == world_size-1)
    {
        localfilas += Ny % world_size;
        printf("Iteraciones %i\n", iteraciones);
        printf("Nx: %i, Ny: %i", Nx, Ny);
    }
    // se creara el vector inicial de la iteracion x de ceros
    float** x= (float**) calloc(Nx, sizeof(float*));
    for (int i = 0; i < Nx; i++)
    {
      x[i] = (float*) calloc(localfilas, sizeof(float));
    }

    printf("\nRank %i, local columnas: %i, first index %i \n", world_rank, localfilas, firstIndex);
    float** const_N = stencil_const_norte(localfilas, firstIndex, Nx, Ny, hx, hy);
    float** const_O = stencil_const_oeste(localfilas, firstIndex, Nx, Ny, hx, hy);
    float** const_S = stencil_const_sur(localfilas, firstIndex, Nx, Ny, hx, hy);
    float** const_E = stencil_const_este(localfilas, firstIndex, Nx, Ny, hx, hy);
    float** const_C = stencil_const_centro(localfilas, firstIndex, Nx, Ny, hx, hy);

    float** f_vec = f_vector(localfilas, firstIndex, Nx);
    float pp = producto_punto(world_rank, world_size, f_vec, f_vec, Nx, localfilas);
    float** Ax = function_mat_vec(world_rank, world_size, const_N, const_O, const_S, 
                const_E, const_C, x, Nx, localfilas);
    float** r = suma_vec(Ax,f_vec, -1, Nx, localfilas);
    float** p; float** z; float** delta_q; float** delta_p; float** beta_p;
    float** q; float rho1; float rho2; float beta; float** error_vec; float error;
    for(int i = 0; i<iteraciones; i++)
    {
      z = r;
      // rho2 sera el rho acutal y 1 es el anterior
      rho2 = producto_punto(world_rank, world_size, r, z, Nx, localfilas);
      if (i == 0){ 
         p = z; // se iguala ya que se considera la matriz identidad
      }
      else{
        beta = rho2/rho1;
        beta_p = esc_vec(p, beta, Nx, localfilas);
        p = suma_vec(z,beta_p, 1, Nx, localfilas);
      }
      q = function_mat_vec(world_rank, world_size, const_N, const_O, const_S, 
                const_E, const_C, p, Nx, localfilas);
      float pt_q = producto_punto(world_rank, world_size, p, q, Nx, localfilas);
      float delta = rho2 / pt_q;
      delta_p = esc_vec(p, delta, Nx, localfilas);
      x = suma_vec(x,delta_p, -1, Nx, localfilas);
      delta_q = esc_vec(q, delta, Nx, localfilas);
      r = suma_vec(r,delta_q, -1, Nx, localfilas);
      if(i%40 == 0 || i == iteraciones-1)
      {
        Ax = function_mat_vec(world_rank, world_size, const_N, const_O, const_S, 
                const_E, const_C, x, Nx, localfilas);
        error_vec = suma_vec(Ax, f_vec, -1, Nx, localfilas);
        error = producto_punto(world_rank, world_size, error_vec, error_vec, Nx, localfilas);
        error = sqrt(error);
        if (world_rank == 0){
          printf("Error en iteracion %i: %f\n", i, error);
        }
      }
      rho1 = rho2;
    }

    Ax = function_mat_vec(world_rank, world_size, const_N, const_O, const_S, 
                const_E, const_C, x, Nx, localfilas);

    //print_matrix(const_N, Nx, localfilas);
    //print_matrix(const_O, Nx, localfilas);
    //print_matrix(const_S, Nx, localfilas);
    //print_matrix(const_E, Nx, localfilas);
    //print_matrix(const_C, Nx, localfilas);
    //print_matrix(Ax, Nx, localfilas);
    //print_matrix(f_vec, Nx, localfilas);
    //print_matrix(x, Nx, localfilas);


    free_matrix(const_N, Nx);
    free_matrix(const_O, Nx);
    free_matrix(const_S, Nx);
    free_matrix(const_E, Nx);
    free_matrix(const_C, Nx);
    free_matrix(f_vec, Nx);
    free_matrix(x, Nx);
    free_matrix(beta_p, Nx);
    free_matrix(Ax, Nx);
    free_matrix(p, Nx);
    free_matrix(r, Nx);
    free_matrix(z, Nx);
    free_matrix(q, Nx);
    // no se hacen los siguentes ya que tienen la misma direccion de memoria que p y q
    free_matrix(delta_p, Nx);
    free_matrix(delta_q, Nx);

	MPI_Finalize();
return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <iostream>
#include <mpi.h>
#include <math.h>
using namespace std;


double alpha(double x, double y)
{
  return x*(x-1)*y*(y-1)+1;
}

void get_local_dom(double** dom, int N_x, int world_rank, int cant_filas)
{
  for (int i=0; i<cant_filas; i++)
  {
    // el -2 porque no guardo los bordes
    dom[i] = (double*) calloc(N_x - 2, sizeof(double*));
    for (int j=0; j<N_x-2; j++)
    {
      dom[i][j] = (double) i;
      // initial guess
    }
  }
}

void get_local_stencil(double** N, double** W, double** C, double** E, double** S, int N_x, int N_y, int world_rank, int world_size, int cant_filas)
{
  double a_1, a_2, a_3, a_4, h_x, h_y;
  int J;
  int prev;
  h_x = (double) 1 / (double) (N_x-1);
  h_y = (double) 1 / (double) (N_y-1);
  if (world_rank != world_size-1)
  {
    prev = world_rank * cant_filas + 1;
  }
  else {prev = ((N_y - 2) / world_size) * world_rank + 1;}
  for (int j=0; j<cant_filas; j++)
  {
    J = prev + j;
    for (int i=0; i<N_x-2; i++)
    {
      a_1 = alpha((i+1)*h_x, (J+0.5)*h_y);
      a_2 = alpha((i+1)*h_x, (J-0.5)*h_y);
      a_3 = alpha((i+0.5)*h_x, J*h_y);
      a_4 = alpha((i+1.5)*h_x, J*h_y);
      // guardo el valor de cada stencil
      N[j][i] = -a_1/(pow(h_y, 2));
      W[j][i] = -a_3/(pow(h_x, 2));
      E[j][i] = -a_4/(pow(h_x, 2));
      S[j][i] = -a_2/(pow(h_y, 2));
      C[j][i] = -(N[j][i] + W[j][i] + E[j][i] + S[j][i]) + 1;
    }
  }
}


void mat_vec(double** N, double** W, double** C, double** E, double** S, double** dom, double** res, int world_rank, int world_size, int local_filas, int N_x)
{
  // como estoy particionando la matriz de forma horizontal, solo debo implementar la comunicación de los nodos norte y sur en los extremos de cada procesador
  double* ultima = (double*) calloc(N_x, sizeof(double));
  double* fila_cero = (double*) calloc(N_x, sizeof(double));
  
  int err;
  MPI_Status status;
  // parto enviando los arrays necesarios: la ultima fila de cada procesador, excepto el ultima, al siguiente:
  if (world_rank != world_size - 1)
  {
    err = MPI_Send(dom[local_filas-1], N_x, MPI_DOUBLE, world_rank+1, world_rank, MPI_COMM_WORLD);
  }

  // recibo la info, que correspondera a la  fila "0" de cada procesador
  if (world_rank != 0)
  {
    err = MPI_Recv(fila_cero, N_x, MPI_DOUBLE, world_rank-1, world_rank-1, MPI_COMM_WORLD, &status);
  }

  // ahora envio la primera fila de cada procesador, excepto el primero, al anterior:
  if (world_rank != 0)
  {
    err = MPI_Send(dom[0], N_x, MPI_DOUBLE, world_rank-1, world_rank, MPI_COMM_WORLD);
  }

  // esto lo recibo como la "ultima" fila de las que le corresponden al procesador
  if (world_rank != world_size - 1)
  {
    err = MPI_Recv(ultima, N_x, MPI_DOUBLE, world_rank+1, world_rank+1, MPI_COMM_WORLD, &status);
  }
 

  // ahora que ya tengo los datos necesarios, procedo a hacer el matvec de manera local
  for (int i=0; i<local_filas; i++)
  {
    for (int j=0; j<N_x; j++)
    {
      res[i][j] = dom[i][j] * C[i][j];

      // sumo el oeste, excepto en el borde
      if (j!=0){res[i][j] += dom[i][j-1] * W[i][j];}

      // sumo el este, excepto en el borde
      if (j!=N_x-1){res[i][j] += dom[i][j+1] * E[i][j];}

      // sumo el norte, con cuidado de no estar en una ultima fila
      if (i != local_filas - 1){res[i][j] += dom[i+1][j] * N[i][j];}
      else if (world_rank != world_size-1){res[i][j] += ultima[j] * N[i][j];}

      // sumo el sur, con cuidado de no estar en la primera fila
      if (i != 0){res[i][j] += dom[i-1][j] * S[i][j];}
      else if (world_rank != 0){res[i][j] += fila_cero[j] * S[i][j];}
    }
  }

  // libero la memoria de los arrays recibidos
  free(ultima);
  free(fila_cero);
}

double dot_prod(double** a, double** b, int world_rank, int world_size, int local_filas, int N_x)
{
  // realiza el producto punto de manera local (multiplico mismas coordenadas y las sumo en un local_sum)
  // y luego hago all_reduce con su suma y guardo esto en la variable res
  double local_prod = 0;
  double result;
  for (int i=0; i<local_filas; i++)
  {
    for (int j=0; j<N_x; j++)
    {
      local_prod += a[i][j] * b[i][j];
    }
  }
  int err;
  err = MPI_Allreduce(&local_prod, &result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return result;
}

void resta_vec(double** a, double** b, double** result, int local_filas, int N_x)
{
  // realiza la resta a-b de manera local
  for (int i=0; i<local_filas; i++)
  {
    for (int j=0; j<N_x; j++)
    {
      result[i][j] = a[i][j] - b[i][j];
    }
  }
}

void igualar(double** a, double** b, int local_filas, int N_x)
{
  // guarda los valores de b en a, manteniendo la existencia e independencia de ambas
  for (int i=0; i<local_filas; i++)
  {
    for (int j=0; j<N_x; j++)
    {
      a[i][j] = b[i][j];
    }
  }
}

void resta_mult(double** a, double** b, double pond, double** res, int local_filas, int N_x)
{
  // hace las lineas del estilo res = a + pond * b (como las dos ultimas del for). Esto tambien se puede hacer de manera local
  for (int i=0; i<local_filas; i++)
  {
    for (int j=0; j<N_x; j++)
    {
      res[i][j] = a[i][j] + pond*b[i][j];
    }
  }
}

void free_matrix(double** a, int local_filas)
{
  for (int i=0; i<local_filas; i++){free(a[i]);}
  free(a);
}

int main()
{
  // cantidad de nodos por eje
  int N_x = 101;
  int N_y = 101;
  
  int iteraciones = 1000;

  // inicializo MPI
  MPI_Init(NULL,NULL);
	int world_size, world_rank;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);


  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);
  printf("Hello world from processor %s, rank %d out of %d processors\n",
          processor_name, world_rank, world_size);


  // lineas 191-201 de la ayudantia del cluster

  int cant_filas = (N_y - 2) / world_size;


  if (world_rank == world_size-1)
  {
    cant_filas += (N_y - 2) % world_size;
  }


  // genero el dominio, que actuará como la incógnita.
  double** dom = (double**) calloc(cant_filas, sizeof(double*));
  get_local_dom(dom, N_x, world_rank, cant_filas);


  // genero la evaluación de f, que actua como el vector b.
  double** f = (double**) calloc(cant_filas, sizeof(double*));
  for (int i=0; i<cant_filas; i++)
  {
    f[i] = (double*) calloc(N_x-2, sizeof(double));
  }


  if (world_rank==world_size/2)
  {
    f[cant_filas/2][N_x/2-1] = (double) 1;
  }

  // genero cada componente del stencil: North, West, Center, East, South
  double** N = (double**) calloc(cant_filas, sizeof(double*));
  for (int i=0; i<cant_filas; i++)
  {
    N[i] = (double*) calloc(N_x-2, sizeof(double));
  }
  double** W = (double**) calloc(cant_filas, sizeof(double*));
  for (int i=0; i<cant_filas; i++)
  {
    W[i] = (double*) calloc(N_x-2, sizeof(double));
  }
  double** C = (double**) calloc(cant_filas, sizeof(double*));
  for (int i=0; i<cant_filas; i++)
  {
    C[i] = (double*) calloc(N_x-2, sizeof(double));
  }
  double** E = (double**) calloc(cant_filas, sizeof(double*));
  for (int i=0; i<cant_filas; i++)
  {
    E[i] = (double*) calloc(N_x-2, sizeof(double));
  }
  double** S = (double**) calloc(cant_filas, sizeof(double*));
  for (int i=0; i<cant_filas; i++)
  {
    S[i] = (double*) calloc(N_x-2, sizeof(double));
  }
  get_local_stencil(N, W, C, E, S, N_x, N_y, world_rank, world_size, cant_filas);

  // genero los "vectores" necesarios para el método
  double** r = (double**) calloc(cant_filas, sizeof(double*));
  for (int i=0; i<cant_filas; i++)
  {
    r[i] = (double*) calloc(N_x-2, sizeof(double));
  }
  double** q = (double**) calloc(cant_filas, sizeof(double*));
  for (int i=0; i<cant_filas; i++)
  {
    q[i] = (double*) calloc(N_x-2, sizeof(double));
  }
  double** p = (double**) calloc(cant_filas, sizeof(double*));
  for (int i=0; i<cant_filas; i++)
  {
    p[i] = (double*) calloc(N_x-2, sizeof(double));
  }

  double** uwu = (double**) calloc(cant_filas, sizeof(double*));
  for (int i=0; i<cant_filas; i++)
  {
    uwu[i] = (double*) calloc(N_x-2, sizeof(double));
  }

  // declaro constantes
  double beta, delta, ro_1, ro_2, res;

  // preparación previa
  mat_vec(N, W, C, E, S, dom, uwu, world_rank, world_size, cant_filas, N_x-2);
  resta_vec(uwu, f, r, cant_filas, N_x-2);
  // uwu es una variable que usa de manera temporal para guardar el matvec Ax antes de restarle f y guardarlo como r

  
  // a iterar!!
  // estoy usando la misma notación que el libro
  for (int i=1; i<iteraciones; i++)
  {
    ro_2 = ro_1;
    //Nota: Como no usamos preacondicionador, los vectores r y z son los mismos.
    ro_1 = dot_prod(r, r, world_size, world_rank, cant_filas, N_x-2);
    // este ro_1 es la norma del residuo, lo imprimimos en el procesador 0
    if (i%(iteraciones/10)==1 || i==2)
    {
      mat_vec(N, W, C, E, S, dom, uwu, world_rank, world_size, cant_filas, N_x-2);
      resta_vec(uwu, f, uwu, cant_filas, N_x-2);
      res = dot_prod(uwu, uwu, world_size, world_rank, cant_filas, N_x-2);
      if (world_rank==0)
      {
        printf("al comenzar la iteración %i, la norma del residuo es %f\n", i, sqrt(res));
      }
    }
    if (i==1)
    {
      igualar(p, r, cant_filas, N_x-2);
    }
    else
    {
      beta = ro_1 / ro_2;
      resta_mult(r, p, beta, p, cant_filas, N_x-2);
    }
    mat_vec(N, W, C, E, S, p, q, world_rank, world_size, cant_filas, N_x-2);
    delta = - ro_1 / dot_prod(p, q, world_rank, world_size, cant_filas, N_x-2);
    resta_mult(dom, p, delta, dom, cant_filas, N_x-2);
    resta_mult(r, q, delta, r, cant_filas, N_x-2);
  }


  //Calculamos e imprimimos la norma del residuo final
  mat_vec(N, W, C, E, S, dom, uwu, world_rank, world_size, cant_filas, N_x-2);
  resta_vec(uwu, f, uwu, cant_filas, N_x-2);
  res = dot_prod(uwu, uwu, world_size, world_rank, cant_filas, N_x-2);
  if (world_rank==0)
  {
    printf("al final, la norma del residuo es %f\n", sqrt(res));
  }


  // a liberar memoria!!
  free_matrix(N, cant_filas);
  free_matrix(W, cant_filas);
  free_matrix(C, cant_filas);
  free_matrix(E, cant_filas);
  free_matrix(S, cant_filas);
  free_matrix(q, cant_filas);
  free_matrix(p, cant_filas);
  free_matrix(r, cant_filas);
  free_matrix(f, cant_filas);
  free_matrix(dom, cant_filas);
  free_matrix(uwu, cant_filas);

  //fin
  MPI_Finalize();
  return 0;
}
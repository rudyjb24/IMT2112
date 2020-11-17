// Compile the code: mpic++ helloMPI.cpp
// Run the executable: mpirun -np 2 ./a.out

#include <iostream>
#include <mpi.h>
using namespace std;

//////////////////////////////////////////IMPORTANTE/////////////////////////////////////////////////////
//Los vectores están en forma de matriz, ya que, es más facil para mi viazualizar los datos, sin embargo
//cumplen con las medidas que deben tener los vectores
//son basicamente un reshape de los vectores originales
/////////////////////////////////////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////IMPORTANTE/////////////////////////////////////////////////////
//El algoritmo está implementado para K = Identidad, 
//por lo que no se ve en el codigo mismo la multiplicación por esta matriz
/////////////////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////IMPORTANTE/////////////////////////////////////////////////////
// * El particionamiento se realiza por columnas
// * Al tener un for i con un for j dentro, el crecimiento de i implica que nos movemos de fila
// y el crecimiento de j implica que nos movemos de columna
/////////////////////////////////////////////////////////////////////////////////////////////////////////

//La funcion alfa calcula el valor de cada punto de la grilla
float alfa(float x, float y)
{
    return x*(x -1)*y*(y-1) +1;
}


//La siguiente funcion genera la matriz de los valores del norte de cada punto
//es decir, le calcula el valor norte del estencil + 1/2 a cada punto
//IMPORTANTE NOTAR EL 1/2, ya que el stencil no evalua en el punto de arriba, sino en el punto mismo más 1/2 hacia arriba
float** matrizN(int primerindice, int Nx, int Ny,float hx, float hy)
{
  float** matrix = (float**) calloc(Nx, sizeof(float*));

  for (int i = 0; i < Nx; i++)
  {
    matrix[i] = (float*) calloc(Ny, sizeof(float));
  }

    for (int i = 0; i < Nx; i++)
  {
      for (int j = 0; j < Ny; j++)
      {   

          matrix[i][j] = - (alfa((i) *hx, (j+ primerindice + 0.5)*hy))/(hy*hy);
      }
  }
  
  return matrix;
}


float** matrizS(int primerindice,int Nx, int Ny,float hx, float hy)
{
    float** matrix = (float**) calloc(Nx, sizeof(float*));

  for (int i = 0; i < Nx; i++)
  {
    matrix[i] = (float*) calloc(Ny, sizeof(float));
  }
  for (int i = 0; i < Nx ; i++)
  {
      for (int j = 0; j < Ny ; j++)
      {   

          matrix[i][j] = - (alfa((i) *hx, (j+ primerindice - 0.5)*hy))/(hy*hy);
      }
  }
  return matrix;
}

float** matrizW(int primerindice,int Nx, int Ny,float hx, float hy)
{
    float** matrix = (float**) calloc(Nx, sizeof(float*));

  for (int i = 0; i < Nx; i++)
  {
    matrix[i] = (float*) calloc(Ny, sizeof(float));
  }
  for (int i = 0; i < Nx; i++)
  {
      for (int j = 0; j < Ny; j++)
      {   

          matrix[i][j] = - (alfa(((i) - 0.5) *hx, (j+ primerindice)*hy))/(hx*hx);
      }
  }
  return matrix;
}

float** matrizE(int primerindice,int Nx, int Ny,float hx, float hy)
{
    float** matrix = (float**) calloc(Nx, sizeof(float*));

  for (int i = 0; i < Nx; i++)
  {
    matrix[i] = (float*) calloc(Ny, sizeof(float));
  }
  for (int i = 0; i < Nx; i++)
  {
      for (int j = 0; j < Ny; j++)
      {   

          matrix[i][j] = - (alfa((( i) + 0.5) *hx, (primerindice +j)*hy))/(hx*hx);
      }
  }
  return matrix;
}

float** matrizC(int primerindice,int Nx, int Ny,float hx, float hy)
{
    float** matrix = (float**) calloc(Nx, sizeof(float*));

  for (int i = 0; i < Nx; i++)
  {
    matrix[i] = (float*) calloc(Ny, sizeof(float));
  }
  for (int i = 0; i < Nx; i++)
  {
      for (int j = 0; j < Ny; j++)
      {   

          matrix[i][j] =  (((alfa(((i )+ 0.5) *hx, (j+ primerindice)*hy)) + (alfa((i - 0.5) *hx, (j+ primerindice)*hy)))/(hx*hx)) +  (((alfa(i*hx, (j + primerindice + 0.5)*hy)) + (alfa((i) *hx, (j + primerindice + 0.5)*hy)))/(hy*hy)) + 1;
      }
  }
  return matrix;
}

void free_matrix(float** matrix, int filas)
{
  for (int i = 0; i < filas; i++)
  {
    free(matrix[i]);
  }
  free(matrix);
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



///////////////////////////////Tips para mat_vec////////////////////////
//Para enviar sin deadlock envio primero los pares y reciben los impares
//luego al reves
//esto con la posicion este y luego con la posicion oeste
/////////////////////////////////////////////////////////////////////////
float** Matriz_vector(int primerindice, int Nx, int Ny, int Ny_original, float** matrizC, float** matrizE, float** matrizW, float** matrizN, float** matrizS, float** v0, int world_rank, int world_size)
{
  float* buffer_oeste = (float*) calloc(Nx, sizeof(float));
  float* buffer_este = (float*) calloc(Nx, sizeof(float));
  float err;
  if (world_rank %2 == 0)
    {
      if (world_rank  != 0)
      {
        err = MPI_Recv(&buffer_oeste, Nx, MPI_FLOAT, world_rank, world_rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        //MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status)  
      }
    }
    else 
    {
      if (world_rank  != world_size -1)
      {
        err = MPI_Send(v0[-1], Nx, MPI_FLOAT, world_rank +1 , world_rank, MPI_COMM_WORLD);
        //printf("Rank %i termino de enviar\n", world_rank);
      }
    }
  
  if (world_rank %2 != 0)
    {
      
      err = MPI_Recv(&buffer_oeste, Nx, MPI_FLOAT, world_rank, world_rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      //MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status)  
    }
    else 
    {
      if (world_rank  != world_size -1)
      {
        err = MPI_Send(v0[-1], Nx, MPI_FLOAT, world_rank +1, world_rank, MPI_COMM_WORLD);
      }
    }
  if (world_rank %2 == 0)
    {
      if (world_rank  != world_size -1)
    {
        err = MPI_Recv(&buffer_este, Nx, MPI_FLOAT, world_rank, world_rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      //MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status) 
    } 
    }
    else 
    {
      if (world_rank  != 0)
      {
        err = MPI_Send(v0[0], Nx, MPI_FLOAT, world_rank - 1 , world_rank, MPI_COMM_WORLD);
        //printf("Rank %i termino de enviar\n", world_rank);
      }
    }
  if (world_rank %2 != 0)
    {
      if (world_rank  != world_size -1)
      {
        err = MPI_Recv(&buffer_este, Nx, MPI_FLOAT, world_rank, world_rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        //MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status) 
      } 
    }
    else 
    {
      if (world_rank  != 0)
      {
        err = MPI_Send(v0[0], Nx, MPI_FLOAT, world_rank - 1, world_rank, MPI_COMM_WORLD);
      }
    }
  
  //Paso 1 multiplicar cada elemento de las 3 matrices(N,S,C) por las posiciones correspondientes en vector v0
  float** result = (float**) calloc(Nx, sizeof(float*));
  for (int i = 0; i < Nx; i++)
    {
      result[i] = (float*) calloc(Ny, sizeof(float));
    }
  for (int i=0; i< Nx; i++)
  {
      for (int j = primerindice; j< Ny; j++)
      {
      //Importante: teoricamenet aqui ya tengo mi super buffer con la información del este y del oeste
        if (i == 0)
        {
          result[i][j] +=  matrizC[i][j]*v0[i][j] + matrizN[i][j]*v0[i-1][j] + matrizE[i][j]*v0[i][j+1] + matrizW[i][j]*v0[i][j-1];
        }
        if (i == Nx)
        {
          result[i][j] += matrizC[i][j]*v0[i][j] + matrizS[i][j]*v0[i+1][j] + matrizE[i][j]*v0[i][j+1] + matrizW[i][j]*v0[i][j-1];
        }
        if (j == 0)
        {
        
          result[i][j] += matrizC[i][j]*v0[i][j] + matrizN[i][j]*v0[i-1][j] + matrizS[i][j]*v0[i+1][j] + matrizW[i][j]*v0[i][j-1];
        }
        if (j == primerindice)
        {
          //Necesito tener la posicion este del procesador anterior
          result[i][j] += matrizC[i][j]*v0[i][j] + matrizN[i][j]*v0[i-1][j] + matrizS[i][j]*v0[i+1][j] + matrizE[i][j]*buffer_este[i] + matrizW[i][j]*v0[i][j-1];
        }
        
        if (j == Ny_original)
        {
          result[i][j] += matrizC[i][j]*v0[i][j] + matrizN[i][j]*v0[i-1][j] + matrizS[i][j]*v0[i+1][j] + matrizE[i][j]*v0[i][j+1];
        }
         
        if (j == Ny)
        {
          //necesito la posicion oeste del procesador siguiente
          result[i][j] += matrizC[i][j]*v0[i][j] + matrizN[i][j]*v0[i-1][j] + matrizS[i][j]*v0[i+1][j] + matrizE[i][j]*v0[i][j+1] + matrizW[i][j]*buffer_oeste[i];
        }
      }
  
  }
  //Paso 4 posicionar la variable en la posicion de la matriz solucion
  free(buffer_oeste);
  free(buffer_este);
  return result;
}

float producto_punto(float** vector1, float** vector2, int Nx, int Ny, int world_rank, int primerindice)
{
  float result = 0;

  for (int i = 0; i < Nx; i++)
  {
    result[i] = (float*) calloc(Ny, sizeof(float));
  }

    for (int i = 0; i < Nx; i++)
  {
      for (int j = primerindice; j < Ny; j++)
      {   

          result +=  vector1[i][j]*vector2[i][j];
      }
  }
  //Envio todos los resultados parciales al procesador 0 y los sumo
  if (world_rank == 0)
    {
      float buffer, float err;
      for (int p=1; p<world_size; p++)
      {
        err = MPI_Recv(&buffer, 1, MPI_INT, p, p, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int i=0; i<n; i++)
        {
          result += buffer;
        }
      }
    }
    else 
    {
      err = MPI_Send(result, 1, MPI_FLOAT, 0, world_rank, MPI_COMM_WORLD);
    }
  //Al tener ya todos los datos en el procesador 0, envio el valor final del resultado a todos los procesadores
  if (world_rank == 0)
    {
      err = MPI_Send(result, 1, MPI_FLOAT, 0, world_rank, MPI_COMM_WORLD);
    }
    else 
    {
      float buffer, float err;
      for (int p=1; p<world_size; p++)
      {
        err = MPI_Recv(&buffer, 1, MPI_INT, p, p, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
    }
  return result;
}

float** ponderacion(float p, float** vector, int Nx, int Ny)
{
  for (int i = 0; i < Nx; i++)
  {
    for (int j = 0; j < Ny; j++)
    {  
      vector[i][j] = p * vector[i][j];
    }
  }
  return vector;
}

//Sirve para suma y resta
float** suma(float** cosa1, float cosa2, int Nx, int Ny, int pon)
{
  float** matrix = (float**) calloc(Nx, sizeof(float*));

  for (int i = 0; i < Nx; i++)
  {
    matrix[i] = (float*) calloc(Ny, sizeof(float));
  }

    for (int i = 0; i < Nx; i++)
  {
      for (int j = 0; j < Ny; j++)
      {   
        if (pon < 0)
        {
          matrix[i][j]  = cosa1[i][j] - cosa2[i][j];
        }
        if (pon > 0)
        {
          matrix[i][j]  = cosa1[i][j] + cosa2[i][j];
        }
      }
  }
  
  return matrix;
}
///////////////////////////IMPORTANTE/////////////////
//como las matrices crecen "hacia abajo" y la grilla hacia arriba el stencil está "dado vuelta"
////S////
//E C W//
////N////

int main()
{
  MPI_Init(NULL,NULL);
  int world_size, world_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  int primerindice, dominiohy, err;

  int Nx = 5; //numero de filas totales
  int Ny = 7; //numero de columnas totales

  float hx = 1/float(Nx);
  float hy = 1/float(Ny);
  

  dominiohy = Ny / world_size;
  primerindice = dominiohy*world_rank;

  if (world_rank == world_size-1)
    {
        dominiohy += Ny % world_size;
    }
  

  printf("Rank %i, local columnas: %i, first index %i \n", world_rank, dominiohy, primerindice);


  //DEFINIMOS TODAS LAS PARTES DE UNA MATRIZ
  float** localMatN= matrizN(primerindice, Nx, dominiohy, hx, hy);
  float** localMatS= matrizS(primerindice,Nx, dominiohy,hx, hy);
  float** localMatE= matrizE(primerindice,Nx, dominiohy, hx, hy);
  float** localMatW= matrizW(primerindice,Nx, dominiohy,hx, hy);
  float** localMatC= matrizC(primerindice,Nx, dominiohy,hx, hy);
  

  //DEFINIREMOS TANTO EL "VECTOR" F  COMO EL VECTOR U COMO MATRICES
  float** vector_f = (float**) calloc(Nx, sizeof(float*));
  for (int i = 0; i < Nx; i++)
  {
    vector_f[i] = (float*) calloc(Ny, sizeof(float));
  }

  float** u_0 = (float**) calloc(Nx, sizeof(float*));
  for (int i = 0; i < Nx; i++)
  {
    u_0[i] = (float*) calloc(Ny, sizeof(float));
  }
  ///////////////Empezamos el algoritmo del gradiente conjugado
  int iter = 10;
  float** p =  Matriz_vector(primerindice, Nx, dominiohy, Ny, localMatC, localMatE, localMatW, localMatN, localMatS, u_0, world_rank, world_size);
  float** p0 = suma(vector_f, p, Nx, dominiohy, -1);

  for (int i = 0; i < iter; i++)
  {
   float** t0 = Matriz_vector(primerindice, Nx, dominiohy,Ny localMatC, localMatE, localMatW, localMatN, localMatS, p0, world_rank, world_size);
   //tenia anotado esto como alfa en el algoritmo, pero en vista de que ya usé alfa le puse alfalfa
   float alfalfa_1 =  producto_punto(p0, p0, Nx, dominiohy, world_rank, primerindice);
   float alfalfa_2 = producto_punto(t0, p0, Nx, dominiohy, world_rank, primerindice);
   float alfalfa = (alfalfa_1* alfalfa_1)/ alfalfa_2;
   float** u_1 = suma(u_0, ponderacion(alfalfa, p0, Nx, dominiohy), Nx, dominiohy, 1);
   float** residual = suma(p0, ponderacion(alfalfa, t0, Nx, dominiohy), Nx, dominiohy, -1);
   float norma_residual = producto_punto(residual, residual, Nx, dominiohy, world_rank, primerindice);
   //beta es la norma del residul, por lo que si este numero deciende el residual converge
   float beta = (norma_residual* norma_residual)/alfalfa_1;
   float** p1 = suma(residual, ponderacion(beta, p0, Nx, dominiohy), Nx, dominiohy, 1);
   float** p0 = p1;
   float** u_0 = u_1;
  }
  print_matrix(u_1, Nx, dominiohy);  

  free_matrix(vector_f, Nx);
  free_matrix(u_0, Nx);
  free_matrix(localMatN, Nx);
  free_matrix(localMatC, Nx);
  free_matrix(localMatS, Nx);
  free_matrix(localMatE, Nx);
  free_matrix(localMatW, Nx);
  MPI_Finalize();
}

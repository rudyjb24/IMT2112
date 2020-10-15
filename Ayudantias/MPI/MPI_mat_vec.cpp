#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <iostream>
#include <mpi.h>
using namespace std;


int** matrix_generator(int filas, int columnas)
{
  int** matrix = (int**) calloc(filas, sizeof(int*));

  for (int i = 0; i < filas; i++)
  {
    matrix[i] = (int*) calloc(columnas, sizeof(int));
  }

  for (int i = 0; i < filas; i++)
  {
    for (int j = 0; j < columnas; j++)
    {
        matrix[i][j] = i+j;
    }
  }
  return matrix;
}

void free_matrix(int **matrix, int filas)
{
  for (int i = 0; i < filas; i++)
  {
    free(matrix[i]);
  }
  free(matrix);
}

void print_matrix(int** matrix, int n)
{
  printf("\n");
  for (int i = 0; i < n; i++) 
  {
      for (int j = 0; j < n; j++)
      {
         printf("%i ", matrix[i][j]); 
      }
    printf("\n");
  }
}

void print_vector(int *vector, int n)
{
  printf("\n");
  for (int i = 0; i < n; i++) 
  {
      printf("%i\n", vector[i]);
  }
}


int main()
{
    MPI_Init(NULL,NULL);
	int world_size, world_rank;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (world_rank == 0)
    {
        // printf("Tamano: %i\n\n", world_size);
    }

    int firstIndex, localColumnas, n, err;
    
    n = 8;

    localColumnas = n / world_size;
    firstIndex = world_rank*localColumnas;

    if (world_rank == world_size-1)
    {
        localColumnas += n % world_size;
    }

    printf("Rank %i, local columnas: %i, first index %i \n", world_rank, localColumnas, firstIndex);

	int localVec[localColumnas];

	for (int i=0; i<localColumnas; i++)
    {
        localVec[i] = firstIndex + i;
	}

    int** localMat = matrix_generator(n, localColumnas);
    print_vector(localVec, localColumnas);
    print_matrix(int** matrix, int n)
	int* localResult = (int*) calloc(n, sizeof(int));

    //printf("Rank %i, empezando local mat vec\n", world_rank);
	for (int i=0; i<n; i++)
    {
        for (int j=0; j<localColumnas; j++)
        {
            localResult[i] += localMat[i][j] * localVec[j];
        }
		
	}
    //printf("Rank %i, termino local mat vec\n", world_rank);

    if (world_rank == 0)
    {
        //printf("Rank 0 va a empezar el proceso de recibir\n");
        int* buffer = (int*) calloc(n, sizeof(int));
        for (int p=1; p<world_size; p++)
        {
            //printf("A punto de recibir de %i\n", p);
            err = MPI_Recv(buffer, n, MPI_INT, p, p, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            //MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status)
            //printf("Recibido de %i\n", p);
            for (int i=0; i<n; i++)
            {
                localResult[i] += buffer[i];
            }
        }
        print_vector(localResult, n);
        free(buffer);
    }
    else 
    {
        //printf("Rank %i va a enviar\n", world_rank);
        err = MPI_Send(localResult, n, MPI_INT, 0, world_rank, MPI_COMM_WORLD);
        //printf("Rank %i termino de enviar\n", world_rank);
        //MPI_Send(void* data, int count, MPI_Datatype datatype, int destination, int tag, MPI_Comm communicator)
    }

    free(localResult);
    free_matrix(localMat, n);
	MPI_Finalize();
}
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <mpi.h>
using namespace std;

/* En teoria esto deberia funcionar para cualquier numero de nodos en la malla y para cualquier 
numero de procesos en MPI*/
// En fin, sorry Rudy por hacerte perder el tiempo u_u

/* 
SENDER 
err = MPI_Send(localResult, n, MPI_INT, 0, world_rank, MPI_COMM_WORLD);
//MPI_Send(void* data, int count, MPI_Datatype datatype, int destination, int tag, MPI_Comm communicator)

RECV
err = MPI_Recv(buffer, n, MPI_INT, p, p, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status)
*/

float dot_local(int* vec1, int* vec2, int loc_size)
{
    int aux=0;
    for(int i=0; i<loc_size; i++)
    {
        aux += vec1[i]*vec2[i];
    }
    return aux;
}

int main()
{
    MPI_Init(NULL,NULL);
	int world_size, world_rank;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int Nx = 3; // parametro
    int Ny = 3; // parametro

    if (world_rank == 0)
    {
        printf("Sanity Check\nTamano de la malla: Nx=%i\tNy=%i",Nx,Ny);
    }

    int err;



    MPI_Finalize();
} 
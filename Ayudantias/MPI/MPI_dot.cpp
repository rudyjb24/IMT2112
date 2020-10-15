#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <iostream>
#include <mpi.h>
using namespace std;


int main()
{
    MPI_Init(NULL,NULL);
	int world_size, world_rank;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (world_rank == 0)
    {
        printf("Tamano: %i\n\n", world_size);
    }

    int localSize, n, err;
    
    n = 16;

    localSize = n / world_size;
    
    printf("Rank %i, local size: %i\n", world_rank, localSize);


    if (world_rank == world_size-1)
    {
        localSize += n % world_size;
    }

	float localVec1[localSize];
	float localVec2[localSize];

	for (int i=0; i<localSize; i++)
    {
		localVec1[i] = 1;
        localVec2[i] = i;
	}


	float localSum = 0;
	for (int i=0; i<localSize; i++)
    {
		localSum += localVec1[i] * localVec2[i];
	}
    printf("Local Sum: %f\n", localSum);
    int globalSum;

    globalSum = 0;
    if (world_rank == 0)
    {
        int receiveBuffer;
        globalSum = localSum;
        for (int p=1; p<world_size; p++)
        {
            err = MPI_Recv(&receiveBuffer, 1, MPI_INT, p, p, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            //MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status)
            globalSum += receiveBuffer;
        }

        printf("El valor es: %f\n", globalSum);
    }
    else 
    {
        err = MPI_Send(&localSum, 1, MPI_INT, 0, world_rank, MPI_COMM_WORLD);
        //MPI_Send(void* data, int count, MPI_Datatype datatype, int destination, int tag, MPI_Comm communicator)
    }

	MPI_Finalize();
}


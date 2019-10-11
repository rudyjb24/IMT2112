#include <iostream>
using namespace std;
#include <mpi.h>

int main(){

    MPI_Init(NULL,NULL);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int firstIndex, localSize;
    int sumlength = 10;



    localSize = sumlength / world_size;
    firstIndex = world_rank * (sumlength/world_size);

    if (world_rank == world_size-1){
        localSize += sumlength % world_size;
    }

     	int localVector[localSize];

        for (int n=0; n<localSize; n++){
                localVector[n] = firstIndex + n;
        }

	int localSum = 0;
        for (int n=0; n<localSize; n++){
                localSum += localVector[n]*localVector[n];
        }

    int globalSum1, globalSum2, globalSum3;

    globalSum1 = 0;
    if (world_rank == 0){
        int receiveBuffer;
        globalSum1 = localSum;
        for (int p=1; p<world_size; p++){
            MPI_Recv(&receiveBuffer, 1, MPI_INT, p, p, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm,$
            globalSum1 += receiveBuffer;
        }
    }
    else {
	MPI_Send(&localSum, 1, MPI_INT, 0, world_rank, MPI_COMM_WORLD);
    //MPI_Send(void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm)
    }

    
    globalSum2 = 0;
    MPI_Reduce(&localSum, &globalSum2, 1, MPI_INT, MPI_SUM, 1, MPI_COMM_WORLD);
    //MPI_Reduce(void* send_data, void* recv_data, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm communicator)

    globalSum3 = 0;
    MPI_Allreduce(&localSum, &globalSum3, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    //MPI_Reduce(void* send_data, void* recv_data, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm communicator)


    if (world_rank == 0)
    {
        printf("Rank 0, primera suma: %i \n", globalSum1);
    }
    else if (world_rank == 1)
    {
        printf("Rank 1, segunda suma: %i \n", globalSum2);
    }
    else if (world_rank == 2)
    {
        printf("Rank 2, tercera suma: %i\n", globalSum3);
    }

    MPI_Finalize();

}

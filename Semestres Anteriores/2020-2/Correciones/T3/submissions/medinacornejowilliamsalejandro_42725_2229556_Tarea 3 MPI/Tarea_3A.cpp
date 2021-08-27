#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <iostream>
#include <mpi.h>
#include <math.h>
using namespace std;


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
         printf("%i ", matrix[i][j]); 
      }
    printf("\n");
  }
}

void print_vector(float* vector, int n)
{
  printf("\n");
  for (int i = 0; i < n; i++) 
  {
      printf("%i\n", vector[i]);
  }
}

void duplicar_matriz(float** matriz_1, float** matriz_2, int world_rank, int world_size, int localFilas, int Nx)
{
     for (int x = 0; x < localFilas; x++)
    {
        for (int j = 0; j < Nx ; j++)
        {   
            matriz_2[x][j] = matriz_1[x][j];
        }
    }
}

float** MPI_vector(int localFilas, float* vector,  int world_rank, int Nx, int Ny)
{   
    int firstIndex = world_rank*localFilas;
    float** X = (float**) calloc(Ny, sizeof(float*));

    for (int i = 0; i < Ny; i++)
    {
        X[i] = (float*) calloc(localFilas, sizeof(float));
    }

    for (int i = firstIndex; i < firstIndex + localFilas; i++)
    {
        for (int j = 0; j < Nx ; j++)
        {   
            X[i][j] = vector[i];
        }
    }
    return X;
}

float** stencil_generator_C(int localFilas, int world_rank, int Nx, int Ny)
{   
    float** C = (float**) calloc(localFilas, sizeof(float*));
    float h_x = 1/Nx ;
    float h_y = 1/Ny ;
    int firstIndex = world_rank * localFilas;

    for (int i = 0; i < localFilas; i++)
    {
        C[i] = (float*) calloc(Ny, sizeof(float));
    }


    for (int x = 0; x < localFilas; x++)
    {
        for (int j = 0; j < Nx ; j++)
        {   
            int i = x + firstIndex + localFilas; 
            C[x][j] = ( ( (i - 1/2) * h_x * ( (i - 1/2) * h_x - 1) * j * h_y * (j * h_y - 1) + 1) + ( (i + 1/2) * h_x * ( (i + 1/2) * h_x - 1) * j * h_y * (j * h_y - 1) + 1) )/pow(h_x,2) + 
                      ( (i * h_x * (i * h_x - 1) * (j - 1/2)* h_y * ( ( j - 1/2) * h_y - 1) + 1) +  (i * h_x * (i * h_x - 1) * (j + 1/2)* h_y * ( ( j + 1/2) * h_y - 1) + 1 ) )/pow(h_y,2) + 
                      1; 
            
        }
    return C;
    }
}

float** stencil_generator_N(int localFilas, int world_rank, int Nx, int Ny)
{
    float** N = (float**) calloc(localFilas, sizeof(float*));
    float h_x = 1/Nx ;
    float h_y = 1/Ny ;
    int firstIndex = world_rank * localFilas;

    for (int i = 0; i < localFilas; i++)
    {
        N[i] = (float*) calloc(Ny, sizeof(float));
    }

    
    for (int x = 0; x < localFilas; x++)
    {
        for (int j = 0; j < Nx ; j++)
        {   
            int i = x + firstIndex + localFilas; 
            N[x][j] = ( -i * h_x * (i * h_x - 1) * (j + 1/2) * h_y * ( (j + 1/2) * h_y - 1) - 1)/pow(h_y, 2);

        }
    }
    return N;
}

float** stencil_generator_S(int localFilas, int world_rank, int Nx, int Ny)
{
    float** S = (float**) calloc(localFilas, sizeof(float*));
    float h_x = 1/Nx ;
    float h_y = 1/Ny ;
    int firstIndex = world_rank * localFilas;

    for (int i = 0; i < localFilas; i++)
    {
        S[i] = (float*) calloc(Ny, sizeof(float));
    }

    
    for (int x = 0; x < localFilas; x++)
    {
        for (int j = 0; j < Nx ; j++)
        {   
            int i = x + firstIndex + localFilas; 
            S[x][j] = ( -i * h_x * (i * h_x - 1) * (j - 1/2) * h_y * ( (j - 1/2) * h_y - 1) - 1)/pow(h_y, 2);

        }
    }
    return S;
}

float** stencil_generator_E(int localFilas, int world_rank, int Nx, int Ny)
{
    float** E = (float**) calloc(localFilas, sizeof(float*));
    float h_x = 1/Nx ;
    float h_y = 1/Ny ;
    int firstIndex = world_rank * localFilas;

    for (int i = 0; i < localFilas; i++)
    {
        E[i] = (float*) calloc(Ny, sizeof(float));
    }

    
    for (int x = 0; x < localFilas; x++)
    {
        for (int j = 0; j < Nx ; j++)
        {   
            int i = x + firstIndex + localFilas; 
            E[x][j] = ( -1 * (i + 1/2) * h_x * ( (i + 1/2) * h_x - 1) * j * h_y * (j * h_y - 1) - 1)/pow(h_x, 2); 

        }
    }
    return E;
}

float** stencil_generator_W(int localFilas, int world_rank, int Nx, int Ny)
{
    float** W = (float**) calloc(localFilas, sizeof(float*));
    float h_x = 1/Nx ;
    float h_y = 1/Ny ;
    int firstIndex = world_rank * localFilas;
    printf("Estamos en el procesador %i ", world_rank);
    for (int i = 0; i < localFilas; i++)
    {
        W[i] = (float*) calloc(Ny, sizeof(float));
    }

    
    for (int x = 0; x < localFilas; x++)
    {
        for (int j = 0; j < Nx ; j++)
        {   
            int i = x + firstIndex + localFilas; 
            W[x][j] = ( -1 * (i - 1/2) * h_x * ( (i - 1/2) * h_x - 1) * j * h_y * (j * h_y - 1) - 1)/pow(h_x, 2);  

        }
    }
    return W;
}

void envio_mensajes(int localFilas, float** vector, int world_rank, int world_size, int Nx, float* ultima_fila_ant, float* primera_fila_sig)
{   //Procesadores pares recibimos y luego enviamos.
    printf("Estamos en el procesador %i ", world_rank);
    float err;
    if (world_rank % 2 == 0)
    {
        if (world_rank < world_size - 1)
        {
            err = MPI_Recv(primera_fila_sig, Nx, MPI_FLOAT, world_rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        if (world_rank > 0)
        {
            err = MPI_Recv(ultima_fila_ant, Nx, MPI_FLOAT, world_rank - 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        if (world_rank < world_size - 1)
        {
            err = MPI_Send(vector[localFilas - 1], Nx, MPI_FLOAT, world_rank + 1, 2, MPI_COMM_WORLD);
        }
        
        if (world_rank > 0)
        {
            err = MPI_Send(vector[0], Nx, MPI_FLOAT, world_rank - 1, 3, MPI_COMM_WORLD);
        }
    }

    //Procesadores impares enviamos y luego recibimos.
    else
    {
        if (world_rank > 0)
        {
            err = MPI_Send(vector[0], Nx, MPI_FLOAT, world_rank - 1, 0, MPI_COMM_WORLD);
        }

        if (world_rank < world_size - 1)
        {
            err = MPI_Send(vector[localFilas - 1], Nx, MPI_FLOAT, world_rank + 1, 1, MPI_COMM_WORLD);
        }

        if (world_rank > 0)
        {
            err = MPI_Recv(ultima_fila_ant, Nx, MPI_FLOAT, world_rank - 1, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        if (world_rank < world_size - 1)
        {
            err = MPI_Recv(primera_fila_sig, Nx, MPI_FLOAT, world_rank + 1, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        }
    }
}

void MPI_mat_cte(float** vector, float cte, int world_rank, int world_size, int Nx, int localFilas)
{ 
    printf("Estamos en el procesador %i ", world_rank);
    float err;
    int n = Nx;
    float* localResult = (float*) calloc(n, sizeof(float));
    int firstIndex = world_rank*localFilas;

    //printf("Rank %i, empezando local mat vec\n", world_rank);
    for (int i = 0; i <  localFilas; i++)
    {
        for (int j = 0; j < Nx ; j++)
        {
            vector[i][j] = vector[i][j] * cte;
        }
	}
    //printf("Rank %i, termino local mat vec\n", world_rank);


    //Acá comienza la comunicación entre procesadores!!!!!!!!!!!!!!!!!
    if (world_rank == 0)
    {
        //printf("Rank 0 va a empezar el proceso de recibir\n");
        float* buffer = (float*) calloc(n, sizeof(float));
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
        //print_vector(localResult, n);
        free(buffer);
    }
    else 
    {
        //printf("Rank %i va a enviar\n", world_rank);
        err = MPI_Send(localResult, n, MPI_INT, 0, world_rank, MPI_COMM_WORLD);
        //printf("Rank %i termino de enviar\n", world_rank);
        //MPI_Send(void* data, int count, MPI_Datatype datatype, int destination, int tag, MPI_Comm communicator)
    }

}

void MPI_mat_vec(float** C, float** N, float** E, float** W, float** S, float** X, int world_rank, int localFilas, int Nx, int Ny,  float* ultima_fila_ant, float* primera_fila_sig)
{   printf("Estamos en el procesador %i ", world_rank);
    for (int i=0; i< Ny; i++)
    {
        for (int j=0; j< Nx; j++)
        {
            float value = 0;
            
            //C_ij   
            value += C[i][j] * X[i][j];

            //S_ij
            if (i == Nx - 1)
            {
                value += S[i][j] * primera_fila_sig[j];
            }
            else 
            {
                value += S[i][j] * X[i+1][j];
            }

            //N_ij
            if (i == 0)
            {
                value += N[i][j] * ultima_fila_ant[j];
            }
            else
            {
                value += N[i][j] * X[i-1][j];
            }

            //E_ij
            value += E[i][j] * X[i][j+1];

            //W_ij
            value += W[i][j] * X[i][j-1];
            
        }

	}
}

float MPI_dot(float** vector_1, float** vector_2, int world_rank, int world_size,  int localFilas, int Nx)
{   printf("Estamos en el procesador %i ", world_rank);
    float err;
    int localSize = localFilas;
    int firstIndex = world_rank*localFilas;
    int localSum = 0;

	for (int i = 0; i <  localFilas; i++)
    {
        for (int j = 0; j < Nx ; j++)
        {
            localSum += vector_1[i][j] * vector_2[i][j];
        }
	}
    //printf("Local Sum: %i\n", localSum);
    
    float globalSum;
    globalSum = 0;
    if (world_rank == 0)
    {
        float receiveBuffer;
        globalSum = localSum;
        for (int p=1; p<world_size; p++)
        {
            err = MPI_Recv(&receiveBuffer, 1, MPI_INT, p, p, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            //MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status)
            globalSum += receiveBuffer;
        }

        //printf("El valor del producto punto es: %i\n", globalSum);
    }
    else 
    {
        err = MPI_Send(&localSum, 1, MPI_INT, 0, world_rank, MPI_COMM_WORLD);
        //MPI_Send(void* data, int count, MPI_Datatype datatype, int destination, int tag, MPI_Comm communicator)
    }   
    return globalSum;
}

void MPI_sum(float** vector_1, float** vector_2, int world_rank, int world_size, int localFilas, int Nx)
{   
    int n = Nx;
    int firstIndex = world_rank*localFilas;
    //printf("Rank %i, empezando local mat vec\n", world_rank);
	for (int i = 0; i <  localFilas; i++)
    {
        for (int j = 0; j < Nx ; j++)
        {
            vector_1[i][j] = vector_1[i][j] + vector_2[i][j];
        }
	}
}

void MPI_suma_matriz(float** vector_1, float** vector_2, int world_rank, int world_size, int localFilas, int Nx)
{   
    printf("Estamos en el procesador %i ", world_rank);
    int n = Nx;
    int firstIndex = world_rank*localFilas;
    //printf("Rank %i, empezando local mat vec\n", world_rank);
	for (int i = 0; i <  localFilas; i++)
    {
        for (int j = 0; j < Nx ; j++)
        {
            vector_1[i][j] = vector_1[i][j] + vector_2[i][j];
        }
	}
    //printf("Rank %i, termino local mat vec\n", world_rank);
}

int main()
{
    MPI_Init(NULL,NULL);
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int firstIndex, localFilas, Nx, Ny, n;
    int n_iteraciones = 100;
    
    Nx = 10;
    Ny = 10;
    
    firstIndex = world_rank * localFilas;

    localFilas = Ny / world_size;

    if (world_rank == world_size-1)
    {
        localFilas += Ny % world_size;
    }
    //printf("firstIndex : %i localColumnas: %i world_rank: %i world_size: %i\n", firstIndex,localColumnas, world_rank, world_size);
    printf("Estamos en el procesador %i ", world_rank);

    float** C = stencil_generator_C(localFilas, world_rank, Nx, Ny);
    float** N = stencil_generator_N(localFilas, world_rank, Nx, Ny);
    float** E = stencil_generator_E(localFilas, world_rank, Nx, Ny);
    float** W = stencil_generator_W(localFilas, world_rank, Nx, Ny);
    float** S = stencil_generator_S(localFilas, world_rank, Nx, Ny);
    float* X_1 = (float*) calloc(localFilas, sizeof(float));
    float** X;
    X = MPI_vector(localFilas, X_1, world_rank, Nx, Ny);
    float* ultima_fila_ant = X[0];
    float* primera_fila_sig = X[localFilas];
    float nuevo_rho, viejo_rho, beta;
    float ** p = (float**) calloc(Ny, sizeof(float*));
    float ** r = (float**) calloc(Ny, sizeof(float*));
    float ** z = (float**) calloc(Ny, sizeof(float*));
    float ** q = (float**) calloc(Ny, sizeof(float*));
    duplicar_matriz(X, r, world_rank, world_size, localFilas, Nx);


    //print_matrix(C, localFilas, Nx);
    
    //Gradiente conjugado
    for (int i = 0; i < n_iteraciones; i++)
    {
        printf("Estamos en la iteración %i\n", i);
        printf("Estamos en el procesador %i\n", world_rank);

        duplicar_matriz(z, r, world_rank, world_size, localFilas, Nx);
        nuevo_rho = MPI_dot(r, r, world_rank, world_size, localFilas, Nx);
        
        if (i==0)
        {
            duplicar_matriz(p, z, world_rank, world_size, localFilas, Nx);
        }

        else
        {
            beta = nuevo_rho/viejo_rho;
            MPI_mat_cte(p, beta, world_rank, world_size, Nx, localFilas);
            MPI_suma_matriz(z, p, world_size, world_size, localFilas, Nx);
        }

        //Comunicamos
        envio_mensajes(localFilas, p, world_rank, world_size, Nx, ultima_fila_ant, primera_fila_sig);

        //Mat_vec Ap
        MPI_mat_vec(C, N, E, W, S, p, world_rank, localFilas, Nx, Ny, ultima_fila_ant, primera_fila_sig);

        //delta
        float punto_p_q = MPI_dot(p, q, world_size, world_size, localFilas, Nx);

        MPI_Allreduce(MPI_IN_PLACE, &punto_p_q, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        float delta = nuevo_rho/punto_p_q;

        MPI_mat_cte(p, delta, world_rank, world_size, Nx, localFilas);

        envio_mensajes(localFilas, X, world_rank, world_size, Nx, ultima_fila_ant, primera_fila_sig);
        viejo_rho = nuevo_rho;

        //La nueva norma 
        MPI_mat_vec(C, N, E, W, S, X, world_rank, localFilas, Nx, Ny, ultima_fila_ant, primera_fila_sig);
        float norma = MPI_dot(q, q, world_rank, world_size, localFilas, Nx);

        MPI_Allreduce(MPI_IN_PLACE, &norma, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    }

    free_matrix(C, Ny);
    free_matrix(N, Ny);
    free_matrix(E, Ny);
    free_matrix(W, Ny);
    free_matrix(S, Ny);
    free_matrix(p, Ny);
    free_matrix(r, Ny);
    free_matrix(z, Ny);
    free_matrix(q, Ny);
    free_matrix(X, Ny);
    free(X_1);
    MPI_Finalize();
}
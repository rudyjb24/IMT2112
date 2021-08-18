#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <chrono>
#include <time.h>
#include <mpi.h>
using namespace std;

float* vector_generator(int length, float range)
{
  float* vector = (float*) calloc(length, sizeof(float));

  for (int i = 0; i < length; i++)
  {
    srand(time(0));
    vector[i]= 2*range*((float)rand()/(float)RAND_MAX) - range;
  }
  
  return vector;
}

void free_matrix(float **matrix, int filas)
{
  for (int i = 0; i < filas; i++)
  {
    free(matrix[i]);
  }
  free(matrix);
}

void print_matrix(int** matrix, int filas, int columnas)
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

void print_vector(float *vector, int n)
{
  printf("\n");
  for (int i = 0; i < n; i++) 
  {
      printf("%i\n", vector[i]);
  }
}

float* mat_vec_stencil(float** stencil, float* vector, int rank, int size, int Nx, int n)
{
    float* new_vector;
    int err;
    float* next_send;
    float* next_recv;
    float* prev_recv;
    float* prev_send;
    if (rank < size-1)
    {
        new_vector = (float*) calloc(n, sizeof(float));
        next_send = (float*) calloc(Nx, sizeof(float));
        next_recv = (float*) calloc(Nx, sizeof(float)); 
    }
    if (0 < rank)
    {
        prev_send = (float*)calloc(Nx, sizeof(float));
        prev_recv = (float*)calloc(Nx, sizeof(float));
    }
    if (rank == size - 1)
    {
        new_vector = (float*) calloc(n, sizeof(float));
    }


    for (int i = 0; i < n; i++)
    {
        new_vector[i] = stencil[2][i] * vector[i];
        int ix = i % Nx;
        if (0 < ix)
        {
            new_vector[i] += stencil[1][i-1] * vector[i-1];
        }
        if (ix < Nx-1)
        {
            new_vector[i] += stencil[3][i+1] * vector[i+1];
        }
        if (Nx <= i)
        {
            new_vector[i] += stencil[0][i-Nx] * vector[i-Nx];
        }
        if (i <= n - 1 - Nx)
        {
            new_vector[i] += stencil[4][i+ Nx] * vector[i+Nx];
        }
        if (rank < size-1)
        {
            if (i > n - 1 - Nx)
            {
                int k = i % (n - Nx);
                next_send[k] += stencil[0][i] * vector[i];
            }
        }
        if (rank > 0)
        {
            if (Nx > i)
            {
                prev_send[i] += stencil[4][i] * vector[i];
            }
        }
    }


    if (rank == 0)
    {
        err = MPI_Send(next_send, Nx, MPI_FLOAT, rank+1, rank, MPI_COMM_WORLD);

        err = MPI_Recv(next_recv, Nx, MPI_FLOAT, rank+1, rank+1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int i = 0; i < Nx; i++)
        {
            new_vector[n-Nx+i] += next_recv[i];
        }
    }

    else
    {
        err = MPI_Recv(prev_recv, Nx, MPI_FLOAT, rank-1, rank-1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if (rank < size-1)
        {
            err = MPI_Send(next_send, Nx, MPI_FLOAT, rank+1, rank, MPI_COMM_WORLD);
            err = MPI_Recv(next_recv, Nx, MPI_FLOAT, rank+1, rank+1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        
        err = MPI_Send(prev_send, Nx, MPI_FLOAT, rank-1, rank, MPI_COMM_WORLD);


        for (int i = 0; i < Nx; i++)
        {
            new_vector[i] += prev_recv[i];
        }

        if (rank < size-1)
        {
            for (int i = 0; i < Nx; i++)
            {
                new_vector[n-Nx+i] += next_recv[i];
            }
        }
      
    }




    return new_vector;
}

float MPI_dot(float* vector1, float* vector2, int n)
{
    int err;
    float localsum = 0;
    float result_sum = 0;
    for (int i=0; i < n; i++)
    {
        localsum += vector1[i]*vector2[i];
    }
    err = MPI_Allreduce(&localsum, &result_sum, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

    return result_sum;
}

float alpha(float x, float y)
{
    int alpha = x * (x-1) * y * (y -1) + 1;
    return alpha;
}

int main()
{
    MPI_Init(NULL,NULL);
	int world_size, world_rank;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    printf("rank: %i out of %i \n", world_rank, world_size);        
    int Nx = 4;
    int Ny = 4;
    int n_iterations = 10;


    int n = Ny / world_size;
    n *= Nx;
    int stepsize = n;
    if (world_rank == world_size - 1)
    {
        n += Nx * ( Ny % stepsize);
    }

    float** stencil = (float**) calloc(5, sizeof(float*));
    for (int i = 0; i < 5; i++)
    {
        stencil[i] = (float*) calloc(n, sizeof(float));
    }
    for (int i = 0; i < n; i++)
    {
        int k_s = i + stepsize * world_rank;
        int x_k = k_s % Nx;
        int y_k = k_s / Nx;
        float x_s = x_k;
        float y_s = y_k;
        if (y_s < Ny - 1)
        {
            stencil[0][i] = (-1) * alpha(x_s, y_s + (1/2)) * Ny * Ny;
        }
        if (0 < y_s)
        {
            stencil[4][i] = (-1) * alpha(x_s, y_s - (1/2)) * Ny * Ny;
        }
        if (x_s < Nx - 1)
        {
            stencil[3][i] = (-1) * alpha(x_s + (1/2), y_s) * Nx * Nx;
        }
        if (0 < x_s)
        {
            stencil[1][i] = (-1) * alpha(x_s - (1/2), y_s) * Nx * Nx;
        }
        
        stencil[2][i] = ((alpha(x_s - (1/2), y_s) + alpha(x_s + (1/2), y_s))* Nx * Nx) + ((alpha(x_s, y_s - (1/2)) + alpha(x_s, y_s + (1/2))) * Ny * Ny) + 1;
    }
    float* f = (float*) calloc(n, sizeof(float));
    for (int i=0; i < n; i++)
    {
        int k_s = i + stepsize * world_rank;
        if (k_s == (Nx * Ny/2) - 1)
        {
            f[i] = 1;
        }
    }
    float* x = vector_generator(n, 0.1);
    float* A_x;
    A_x = mat_vec_stencil(stencil, x, world_rank, world_size, Nx, n);


    float* r1 = (float*) calloc(n, sizeof(float));
    float rho_1, beta, rho_2, delta;
    float* p;
    float* q;
    float* r2;
    for (int i=0; i<n; i++)
    {
        r1[i] = A_x[i] - f[i];
    }
    for (int k=0; k<n_iterations; k++)
    {
        rho_1 = MPI_dot(r1, r1, n);
        printf("residuo nuevo en iteración %i vale %f \n", k, rho_1);

        if (k == 0)
        {
            p = r1;
        }

        else
        {
            float beta = rho_1/rho_2;
            for(int i=0; i<n; i++)
            {
                p[i] = r1[i] + beta * p[i];
            }


        }
        rho_2 = rho_1;
        r2 = r1;
        q = mat_vec_stencil(stencil, p, world_rank, world_size, Nx, Ny);
        delta = rho_2 / MPI_dot(q, p, n);
        for (int i=0; i<n; i++)
        {
            x[i] -= delta * p[i];
            r1[i] -= delta * q[i];
        }
        if (MPI_dot(r1, r1, n) >= rho_1)
        {
            printf("no converge :( \n");
            break;
        }
        

    }
    free_matrix(stencil, 5);

    MPI_Finalize();
    return 0;




}
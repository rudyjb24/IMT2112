#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include<cmath>
#include <mpi.h>
using namespace std;

float alpha(float x, float y)
{
    return x*(x - 1)*y*(y - 1) + 1;
}

float** stencil(int Nx, int Ny, int firstIndex, int localRows, int l) //Genera el senticl del nodo l -> (i,j)
{
    int i = l%localRows;
    int j = l/localRows;

    i += firstIndex + 1;
    j += 1;

    float hx,hy,ai0,ai1,aj0,aj1;
    hx = 1/Nx;
    hy = 1/Ny;
    ai0 = alpha((i-0.5)*hx,j*hy);
    ai1 = alpha((i+0.5)*hx,j*hy);
    aj0 = alpha(i*hx,(j-0.5)*hy);
    aj1 = alpha(i*hx,(j+0.5)*hy);

    float** sten = (float**) calloc(3, sizeof(float*));
    for (int i = 0; i < 3; i++)
    {
        sten[i] = (float*) calloc(3, sizeof(float));
    }
    sten[0][1] = -aj1*(pow(Ny, 2));
    sten[1][0] = -ai0*(pow(Nx, 2));
    sten[1][1] = (ai0 + ai1)*(pow(Nx, 2)) + (aj0 + aj1)*(pow(Ny, 2)) + 1;
    sten[1][2] = -ai1*(pow(Nx, 2));
    sten[2][1] = -aj0*(pow(Ny, 2));

    return sten;
}

float dot(float* u, float* v, int localSize, MPI_Comm comm)
{
    float localDot = 0.0;
    for (int i = 0; i < localSize; i++)
    {
        localDot += u[i]*v[i];
    }
    float dot;
    MPI_Allreduce(&localDot, &dot, 1, MPI_FLOAT, MPI_SUM, comm);
    return dot;
}

float* comb(float a, float* u, float* v, int localSize) //u + av
{
    for (int i = 0; i < localSize; i++)
    {
        u[i] += a*v[i];
    }
    return u;
}

float* matvec(float* v, float*** A, int Ny, int localSize, int localRows, int firstIndex, int world_rank, int world_size, MPI_Comm comm)
{
    float* y = (float*) calloc(localSize, sizeof(float));

    float* vleft = (float*) calloc(Ny - 1, sizeof(float));

    float* vright = (float*) calloc(Ny - 1, sizeof(float));
    vleft[0] = v[0];
    for (int i = 1; i < Ny - 1; i++)
    {
        vleft[i] = v[i*localRows];
        vright[i - 1] = v[i*localRows - 1];
    }
    vright[Ny - 2] = v[localSize - 1];

    float* leftedge = (float*) calloc(Ny - 1, sizeof(float));
    float* rightedge = (float*) calloc(Ny - 1, sizeof(float));
    if (world_rank == 0)
    {
        MPI_Send(vright, Ny - 1, MPI_FLOAT, world_rank + 1, world_rank + 1, comm);
        MPI_Recv(rightedge, Ny - 1, MPI_FLOAT, world_rank + 1, world_rank + 1, comm, MPI_STATUSES_IGNORE);
    }
    if (world_rank == world_size)
    {
        MPI_Recv(leftedge, Ny - 1, MPI_FLOAT, world_rank - 1, world_rank - 1, comm, MPI_STATUSES_IGNORE);
        MPI_Send(vleft, Ny - 1, MPI_FLOAT, world_rank - 1, world_rank - 1, comm);
    }
    else
    {
        MPI_Recv(leftedge, Ny - 1, MPI_FLOAT, world_rank - 1, world_rank - 1, comm, MPI_STATUSES_IGNORE);
        MPI_Send(vright, Ny - 1, MPI_FLOAT, world_rank + 1, world_rank + 1, comm);
        MPI_Recv(rightedge, Ny - 1, MPI_FLOAT, world_rank + 1, world_rank + 1, comm, MPI_STATUSES_IGNORE);
        MPI_Send(vleft, Ny - 1, MPI_FLOAT, world_rank - 1, world_rank - 1, comm);
    }
    free(vleft);
    free(vright);

    for (int i = 0; i < localSize; i++)
    {
        y[i] = A[i][1][1]*v[i];
        if (localRows == 1)
        {
            y[i] += A[i][1][0]*leftedge[i] + A[i][1][2]*rightedge[i];
            if (i/localRows == 0)
            {
                y[i] += A[i][0][1]*v[i + localRows];
            }
            if (i/localRows == Ny - 1)
            {
                y[i] += A[i][2][1]*v[i - localRows];
            }
            else
            {
                y[i] += A[i][0][1]*v[i + localRows] + A[i][2][1]*v[i - localRows];
            }
        }
        else
        {
            if (i/localRows == 0)
            {
                y[i] += A[i][0][1]*v[i + localRows];
                if (i == 0)
                {
                    y[i] += A[i][1][0]*leftedge[i] + A[i][1][2]*v[i + 1];
                }
                if (i == localRows - 1)
                {
                    y[i] += A[i][1][0]*v[i - 1] + A[i][1][2]*rightedge[i];
                }
                else
                {
                    y[i] += A[i][1][0]*v[i - 1] + A[i][1][2]*v[i + 1];
                }
            }
            if (i/localRows == Ny - 1)
            {
                y[i] += A[i][2][1]*v[i - localRows];
                if (i == localSize - localRows + 1)
                {
                    y[i] += A[i][1][0]*leftedge[Ny - 1] + A[i][1][2]*v[i + 1];
                }
                if (i == localSize - 1)
                {
                    y[i] += A[i][1][0]*v[i - 1] + A[i][1][2]*rightedge[Ny - 1];
                }
                else
                {
                    y[i] += A[i][1][0] * v[i - 1] + A[i][1][2] * v[i + 1];
                }
            }
            if (i%localRows == 0)
            {
                if (i == 0)
                {
                    y[i] = y[i];
                }
                if (i == localSize - localRows + 1)
                {
                    y[i] = y[i];
                }
                else
                {
                    y[i] += A[i][1][0]*leftedge[i/localRows] + A[i][1][2]*v[i + 1] + A[i][0][1]*v[i + localRows] + A[i][2][1]*v[i - localRows];
                }
            }
            if (i%localRows == localRows - 1)
            {
                if (i == localRows - 1)
                {
                    y[i] = y[i];
                }
                if (i == localSize - 1)
                {
                    y[i] = y[i];
                }
                else
                {
                    y[i] += A[i][1][0]*v[i - 1] + A[i][1][2]*rightedge[i/localRows] + A[i][0][1]*v[i + localRows] + A[i][2][1]*v[i - localRows];
                }
            }
            else
            {
                y[i] += y[i] += A[i][1][0]*v[i - 1] + A[i][1][2]*v[i + 1] + A[i][0][1]*v[i + localRows] + A[i][2][1]*v[i - localRows];
            }
        }
    }
    free(leftedge);
    free(rightedge);
    return y;
}

int test_error(float* r, int localSize, MPI_Comm comm, float error)
{
    float norm_r = pow(dot(r, r, localSize, comm), 0.5);
    if (norm_r < error)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

float*** generate_stens(int localRows, int firstIndex, int Nx, int Ny)
{
    int localSize = localRows*(Ny-1);
    float*** stens = (float***) calloc(localRows*(Ny-1), sizeof(float**));
    for (int l = 0; l < localSize; l++)
    {
        stens[l] = stencil(Nx, Ny, firstIndex, localRows, l);
    }
    return stens;
}

void free_stencils(float*** stens, int localSize)
{
    for (int i = 0; i < localSize; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            free(stens[i][j]);
        }
        free(stens[i]);
    }
    free(stens);
}

int main()
{
    MPI_Init(NULL,NULL);
    MPI_Comm comm = MPI_COMM_WORLD;
	int world_size, world_rank;
	MPI_Comm_size(comm, &world_size);
	MPI_Comm_rank(comm, &world_rank);

    int Nx = 10;
    int Ny = Nx;

	int firstIndex, localRows, localSize, err;

    localRows = (Nx-1)/world_size; //Dividimos los nodos por columnas
    firstIndex = world_rank*localRows;
    if (world_rank == world_size-1)
    {
        localRows += (Nx-1)%world_size;
    }

    localSize = localRows*(Ny-1);

    float* r = (float*) calloc(localSize, sizeof(float));
    float* x = r;
    if (world_rank == 0)
    {
        r[0] = 1;
    }
    float* d = r;

    float*** A = generate_stens(localRows, firstIndex, Nx, Ny);

    float* Ad = (float*) calloc(localSize, sizeof(float));
    float alpha, beta, rdotr, rdotr2;
    float error = 0.01;

    rdotr = dot(r, r, localSize, comm);

    for (int i = 1; i < 300; i++)
    {
        Ad = matvec(d, A, Ny, localSize, localRows, firstIndex, world_rank, world_size, comm);
        alpha = rdotr/dot(d, Ad, localSize, comm);
        x = comb(alpha, x, d, localSize);
        r = comb(-alpha, r, Ad, localSize);
        if (test_error(r, localSize, comm, error) == 1)
        {
            i = 300;
        }
        rdotr2 = dot(r, r, localSize, comm);
        beta = rdotr/rdotr2;
        rdotr = rdotr2;
        d = comb(beta, r, d, localSize);
    }

    free(x);
    free(d);
    free(r);
    free(Ad);
    free_stencils(A, localSize);
    MPI_Finalize();
}

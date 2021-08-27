#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h> 
#include <mpi.h>
using namespace std;



float local_dot_product(float *vector1, float *vector2, int n)
{
    float result = 0.0;
    int i = 0;
    for (int i=0; i < n; i++){ 
      result += (vector1[i] * vector2[i]);
    }
    return result;
}


float alpha(float x, float y){
    float result;
    result = (x * (x - 1) * y * (y - 1)) + 1;
    return result;
}


int main(){
    MPI_Init(NULL, NULL);
	int world_size, world_rank;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int Nx = 100;
    int Ny = 100;
    float hx = 1 / (float)(Nx - 1);
    float hy = 1 / (float)(Ny - 1);


    if (world_rank == 0){
        printf("Corriendo CG para diferencias finitas con %i procesadores \n", world_size);
    }
    int firstIndex, localSize, err, localRows;
    
    localRows = Ny / world_size;
    firstIndex = localRows * Nx * world_rank;

    if (world_rank == world_size-1)
    {
        localRows += (Ny % world_size);
    }
    localSize = localRows * Nx;
    float** stencil = (float**) calloc(localSize, sizeof(float*));

    int i = 0;
    int j = firstIndex / Nx;
    for (int n = 0; n < localSize; n++)
    {
        stencil[n] = (float*) calloc(5, sizeof(float));
        stencil[n][0] = (-1 / (hy * hy)) * alpha((float)(i * hx), (float)(j - 0.5) * hy );
        stencil[n][1] = (-1 / (hx * hx)) * alpha((float)(i - 0.5) * hx, j * hy );
        stencil[n][3] = (-1 / (hx * hx)) * alpha((float)(i + 0.5) * hx, j * hy );
        stencil[n][4] = (-1 / (hy * hy)) * alpha( i * hx, (float)(j + 0.5) * hy );
        stencil[n][2] = -(stencil[n][0] + stencil[n][1] + stencil[n][3] + stencil[n][4]) + 1;
        // printf("%f ", stencil[n][0]);
        // printf("\n");

        i += 1;
        if (i % Nx == 0 ){
            i = 0;
            j += 1;
        }
    }
    i = 0;
    j = firstIndex / Nx;

    // Inicializamos x como vector nulo 

    float* x = (float *)calloc(localSize, sizeof(float));
    for (int k = 0; k < localSize; k++) 
    {
        x[k] = 0;
    }

    // b = f, con f en el nodo (0,0) valiendo 1 
    // Como x es nulo inicialmente, r = -b
    float* b = (float *)calloc(localSize, sizeof(float));
    float* r = (float *)calloc(localSize, sizeof(float));
    for (int k = 0; k < localSize; k++) 
    {
        if (k == 0 && world_rank == 0)
        {
            b[k] = 1;
        } else
        {
            b[k] = 0;
        }
        r[k] = -b[k];
    }    
    // Inicializamos los vectores y valores que necesitaremos
    float* p = (float *)calloc(localSize, sizeof(float));
    float* q = (float *)calloc(localSize, sizeof(float));
    float rho_0;
    float rho_1;
    float beta;
    float delta;
    float* p_recv_down = (float *)calloc(Nx, sizeof(float));
    float* p_recv_up = (float *)calloc(Nx, sizeof(float));
    float localErr;
    float totalErr;

	int max_iter = 200;
    // Ahora comenzamos el for loop
    for (int n=1; n<=max_iter; n++)
    {
        rho_1 = 0;
        float local_rho = local_dot_product(r, r, localSize);
        MPI_Allreduce(&local_rho, &rho_1, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        if (n == 1){
            for(int k = 0; k < localSize; k++){
                p[k] = r[k];
            }
        }else{
            beta = rho_1 / rho_0;
            for (int k = 0; k < localSize; k++)
            {
                p[k] = r[k] + (beta * p[k]);
            }    
        }
        // printf("valor p 0 : %f \n", p[0]);
        // Ahora calculamos q = Ap usando los stencils. Primero tendremos que comunicar los valores de p
        // Los procesadores enviarán primero su información hacia arriba, a excepción del último:
        if (world_rank != world_size - 1){
            float* p_send_up = (float *)calloc(Nx, sizeof(float));
            for(int k = 0; k < Nx; k++){
                p_send_up[k] = p[localSize - Nx + k];
            }
            // printf("Procesador %i, enviando %f, %f, %f a procesador %i \n", world_rank, p_send_up[0], p_send_up[1], p_send_up[2], world_rank + 1);
            err = MPI_Send(p_send_up, Nx, MPI_FLOAT, world_rank + 1, 0, MPI_COMM_WORLD);

        }
        // Tenemos que recibir la información DESDE ABAJO de manera ordenada, todos reciben a excepción del primero
        if(world_rank != 0){
            err = MPI_Recv(p_recv_down, Nx, MPI_FLOAT, world_rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // printf("Procesador %i, recibido %f, %f, %f desde proce %i \n", world_rank, p_recv_down[0], p_recv_down[1], p_recv_down[2], world_rank - 1);
        }
        // Ahora enviamos la información hacia abajo
        if (world_rank > 0){
            float* p_send_down = (float *)calloc(Nx, sizeof(float));
            for(int k = 0; k < Nx; k++){
                p_send_down[k] = p[k];
            }
            // printf("Procesador %i, enviando %f, %f, %f a procesador %i \n", world_rank, p_send_down[0], p_send_down[1], p_send_down[2], world_rank - 1);
            err = MPI_Send(p_send_down, Nx, MPI_FLOAT, world_rank - 1, 1, MPI_COMM_WORLD);
        }
        if (world_rank != world_size - 1){
            err = MPI_Recv(p_recv_up, Nx, MPI_FLOAT, world_rank + 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // printf("Procesador %i, recibido %f, %f, %f desde proce %i \n", world_rank, p_recv_up[0], p_recv_up[1], p_recv_up[2], world_rank + 1);
        } 
        // Con el bloque anterior ya tenemos los valores de los nodos que necesitamos
        i = 0;  // partimos desde el nodo de la izquierda
        j = firstIndex / Nx; // partimos desde lo más abajo de la malla
        float* node_stencil = (float *)calloc(5, sizeof(float));
        for(int relative_index = 0; relative_index < localSize; relative_index++){
            // Aquí tenemos que identificar los puntos que están sobre los bordes
            // de la malla y sobre los bordes de los bloques de filas
            // Recordar que queremos calcular q = Ap
            node_stencil = stencil[relative_index];
            // Siempre tenemos el centro del stencil
            q[relative_index] = node_stencil[2] *  p[relative_index];
            // Hay que incluir el Sur sólo cuando j > 0
            if (j > 0){
                // Esta p_{i, j - 1} en el procesador?
                // La primera fila del procesador es firstIndex/Nx
                if(j == firstIndex / Nx){
                    // Tenemos que buscar en el procesador de abajo
                    q[relative_index] += node_stencil[0] * p_recv_down[i];
                }else{
                    // En este caso el Nodo sur está en el mismo procesador
                    q[relative_index] += node_stencil[0] * p[relative_index - Nx];
                }
            }
            // Hay que incluir el Oeste sólo cuando i > 0
            // Como la partición es por filas está en el mismo procesador.
            if (i > 0){
                q[relative_index] += node_stencil[1] * p[relative_index - 1];
            }
            // Hay que incluir el Este sólo cuando i < Nx - 1
            // Como la partición es por filas está en el mismo procesador.
            if (i < Nx - 1){
                q[relative_index] += node_stencil[3] * p[relative_index + 1];
            }
            // Hay que incluir el Norte sólo cuando j < Ny - 1
            if (j < Ny - 1){
                // Esta p_{i, j + 1} en el procesador?
                // La última fila del procesador es
                if( relative_index >= localSize - Nx ){
                    // Tenemos que buscar en el procesador de arriba
                    q[relative_index] += node_stencil[4] * p_recv_up[i];
                }else{
                    // En este caso el Norte está en el mismo procesador
                    q[relative_index] += node_stencil[4] * p[relative_index + Nx];
                }
            }
            i += 1;
            if (i % Nx == 0 ){
                i = 0;
                j += 1;
            }
        }
        // Con q calculado seguimos la iteración
    
        float dot_pq = 0;
        float local_pq = local_dot_product(p, q, localSize);
        MPI_Allreduce(&local_pq, &dot_pq, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

        delta = rho_1 / dot_pq;
        // printf("delta: %f en proce %i  iteracion %i\n", dot_pq, world_rank, n);
        for(int k = 0; k < localSize; k++){
            x[k] =  x[k] - (delta * p[k]); 
            r[k] = r[k] - (delta * q[k]);      
        }

        if(world_rank == 0){
            if(n % 10 == 0 || n==1){ 
                printf("Error en iteración %i: %f \n", n, sqrt(rho_1));
            }
        }
        rho_0 = rho_1;
    }
	MPI_Finalize();
}


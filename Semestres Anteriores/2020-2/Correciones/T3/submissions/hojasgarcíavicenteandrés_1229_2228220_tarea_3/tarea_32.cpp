#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
using namespace std;

float** N_generator(int Nx, int Ny, int world_rank, int world_size) {
    // Tamaño de los bloques de nodos
    int block_size = Nx/world_size;
    if (world_rank == world_size - 1) {
        block_size = Nx/world_size + Nx%world_size;
    }
    float** array = (float**) calloc(block_size, sizeof(float*));
    for (int i=0; i<block_size; i++) {
        array[i] = (float*) calloc(Ny, sizeof(float));
    }
    // primer indice de la particion en el eje x
    int first_index = (Nx/world_size)*world_rank;
    for (int i=0; i<block_size; i++) {
        for (int j=0; j<Ny; j++) {
            if (j == Ny-1) {
                array[i][j] = 0;
            } else {
                // usamos alpha = x(x-1)y(y-1)
                float hx = 1.0/(float)(Nx+1.0);
                float hy = 1.0/(float)(Ny+1.0);
                float x = ((float) i + first_index + 1.0)*hx;
                float y = ((float) j + 1.0 + 0.5)*hy;
                array[i][j] = -((x*(x-1.0)*y*(y-1.0) + 1.0)/(hy*hy));
            }
        }
    }
    return array;
}

float** S_generator(int Nx, int Ny, int world_rank, int world_size) {
    // Tamaño de los bloques de nodos
    int block_size = Nx/world_size;
    if (world_rank == world_size - 1) {
        block_size = Nx/world_size + Nx%world_size;
    }
    float** array = (float**) calloc(block_size, sizeof(float*));
    for (int i=0; i<block_size; i++) {
        array[i] = (float*) calloc(Ny, sizeof(float));
    }
    // primer indice de la particion en el eje x
    int first_index = (Nx/world_size)*world_rank;
    for (int i=0; i<block_size; i++) {
        for (int j=0; j<Ny; j++) {
            if (j == 0) {
                array[i][j] = 0;
            } else {
                // usamos alpha = x(x-1)y(y-1)
                float hx = 1.0/(float)(Nx+1.0);
                float hy = 1.0/(float)(Ny+1.0);
                float x = ((float) i + first_index + 1.0)*hx;
                float y = ((float) j + 1.0 - 0.5)*hy;
                array[i][j] = -((x*(x-1.0)*y*(y-1.0) + 1.0)/(hy*hy));
            }
        }
    }
    return array;
}

float** W_generator(int Nx, int Ny, int world_rank, int world_size) {
    // Tamaño de los bloques de nodos
    int block_size = Nx/world_size;
    if (world_rank == world_size - 1) {
        block_size = Nx/world_size + Nx%world_size;
    }
    float** array = (float**) calloc(block_size, sizeof(float*));
    for (int i=0; i<block_size; i++) {
        array[i] = (float*) calloc(Ny, sizeof(float));
    }
    // primer indice de la particion en el eje x
    int first_index = (Nx/world_size)*world_rank;
    for (int i=0; i<block_size; i++) {
        for (int j=0; j<Ny; j++) {
            if (world_rank == 0 && i == 0) {
                array[i][j] = 0;
            } else {
                // usamos alpha = x(x-1)y(y-1)
                float hx = 1.0/(float)(Nx+1.0);
                float hy = 1.0/(float)(Ny+1.0);
                float x = ((float) i + first_index + 1.0 - 0.5)*hx;
                float y = ((float) j + 1.0)*hy;
                array[i][j] = -((x*(x-1.0)*y*(y-1.0) + 1.0)/(hx*hx));
            }
        }
    }
    return array;
}

float** E_generator(int Nx, int Ny, int world_rank, int world_size) {
    // Tamaño de los bloques de nodos
    int block_size = Nx/world_size;
    if (world_rank == world_size - 1) {
        block_size = Nx/world_size + Nx%world_size;
    }
    float** array = (float**) calloc(block_size, sizeof(float*));
    for (int i=0; i<block_size; i++) {
        array[i] = (float*) calloc(Ny, sizeof(float));
    }
    // primer indice de la particion en el eje x
    int first_index = (Nx/world_size)*world_rank;
    for (int i=0; i<block_size; i++) {
        for (int j=0; j<Ny; j++) {
            if (world_rank == world_size-1 && i == block_size-1) {
                array[i][j] = 0;
            } else {
                // usamos alpha = x(x-1)y(y-1)
                float hx = 1.0/(float)(Nx+1.0);
                float hy = 1.0/(float)(Ny+1.0);
                float x = ((float) i + first_index + 1.0 + 0.5)*hx;
                float y = ((float) j + 1.0)*hy;
                array[i][j] = -((x*(x-1.0)*y*(y-1.0) + 1.0)/(hx*hx));
            }
        }
    }
    return array;
}

float** C_generator(int Nx, int Ny, int world_rank, int world_size) {
    // Tamaño de los bloques de nodos
    int block_size = Nx/world_size;
    if (world_rank == world_size - 1) {
        block_size = Nx/world_size + Nx%world_size;
    }
    float** array = (float**) calloc(block_size, sizeof(float*));
    for (int i=0; i<block_size; i++) {
        array[i] = (float*) calloc(Ny, sizeof(float));
    }
    // primer indice de la particion en el eje x
    int first_index = (Nx/world_size)*world_rank;
    for (int i=0; i<block_size; i++) {
        for (int j=0; j<Ny; j++) {
            if (world_rank == world_size-1 && i == block_size-1) {
                array[i][j] = 0;
            } else {
                // usamos alpha = x(x-1)y(y-1)
                float hx = 1.0/(float)(Nx+1.0);
                float hy = 1.0/(float)(Ny+1.0);
                // North
                float x = ((float) i + first_index + 1.0)*hx;
                float y = ((float) j + 1.0 + 0.5)*hy;
                array[i][j] += ((x*(x-1.0)*y*(y-1.0) + 1.0)/(hy*hy));
                // South
                x = ((float) i + first_index + 1.0)*hx;
                y = ((float) j + 1.0 - 0.5)*hy;
                array[i][j] += ((x*(x-1.0)*y*(y-1.0) + 1.0)/(hy*hy));
                // West
                x = ((float) i + first_index + 1.0 - 0.5)*hx;
                y = ((float) j + 1.0)*hy;
                array[i][j] += ((x*(x-1.0)*y*(y-1.0) + 1.0)/(hx*hx));
                // East
                x = ((float) i + first_index + 1.0 + 0.5)*hx;
                y = ((float) j + 1.0)*hy;
                array[i][j] += ((x*(x-1.0)*y*(y-1.0) + 1.0)/(hx*hx));
                array[i][j] += 1.0;
            }
        }
    }
    return array;
}

float** random_vector(int Nx, int Ny, int world_rank, int world_size) {
    int range = 100;
    int block_size = Nx/world_size;
    if (world_rank == world_size-1) {
        block_size = Nx/world_size + Nx%world_size;
    }
    float** array = (float**) calloc(block_size, sizeof(float*));
    for (int i=0; i<block_size; i++) {
        array[i] = (float*) calloc(Ny, sizeof(float));
    }
    for (int i=0; i<block_size; i++) {
        for (int j=0; j<Ny; j++) {
            array[i][j] = 2*range*((float)rand()/(float)RAND_MAX) - range;
        }
    }
    return array;
}

float** function_vector(int Nx, int Ny, int world_rank, int world_size) {
    int block_size = Nx/world_size;
    if (world_rank == world_size-1) {
        block_size = Nx/world_size + Nx%world_size;
    }
    float** array = (float**) calloc(block_size, sizeof(float*));
    for (int i=0; i<block_size; i++) {
        array[i] = (float*) calloc(Ny, sizeof(float));
    }
    for (int i=0; i<block_size; i++) {
        for (int j=0; j<Ny; j++) {
            // se elige una fuente puntual
            if (i == block_size/2 && j == Ny/2 && world_rank == 0) {
                array[i][j] = 1.0;
            }
        }
    }
    return array;
}

void free_matrix(float** matrix, int n) {
    for (int i=0; i<n; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

float** vec_diff(float** a, float** b, int n, int m) {
    // vector difference a-b
    float** result = (float**) calloc(n, sizeof(float*));
    for (int i=0; i<n; i++) {
        result[i] = (float*) calloc(m, sizeof(float));
    }
    for (int i=0; i<n; i++) {
        for (int j=0; j<m; j++) {
            result[i][j] = a[i][j] - b[i][j];
        }
    }
    return result;
}

float** mat_vec(float** N, float** S, float** E, float** W, float** C, float** x, 
                int Nx, int Ny, int world_rank, int world_size) {
    // matrix vector multiplication
    int err;
    // elementos con x_first_index-1
    float* buffer_left = (float*) calloc(Ny, sizeof(float));
    // elementos con x_first_index + block_size
    float* buffer_right = (float*) calloc(Ny, sizeof(float));
    int block_size = Nx/world_size;
    if (world_rank == world_size-1) {
        block_size = Nx/world_size + Nx%world_size;
    }

    // variable resultado
    float** result = (float**) calloc(block_size, sizeof(float*));
    for (int i=0; i<block_size; i++) {
        result[i] = (float*) calloc(Ny, sizeof(float));
    }

    // COMUNICACION
    // Comunicacion entre procesadores, procesador par envía y procesador 
    // impar recibe
    if (world_rank%2 == 0) {
        // Pares envian primero y reciben despues
        if (world_rank == 0) {
            err = MPI_Send(x[block_size-1], Ny, MPI_FLOAT, world_rank+1, 0, MPI_COMM_WORLD);
            err = MPI_Recv(buffer_right, Ny, MPI_FLOAT, world_rank+1, world_rank+1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else if (world_rank == world_size-1) {
            err = MPI_Send(x[0], Ny, MPI_FLOAT, world_rank-1, world_rank, MPI_COMM_WORLD);
            err = MPI_Recv(buffer_left, Ny, MPI_FLOAT, world_rank-1, world_rank-1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else {
            // Procesadores pares no extremos
            err = MPI_Send(x[0], Ny, MPI_FLOAT, world_rank-1, world_rank, MPI_COMM_WORLD);
            err = MPI_Send(x[block_size-1], Ny, MPI_FLOAT, world_rank+1, world_rank, MPI_COMM_WORLD);
            err = MPI_Recv(buffer_left, Ny, MPI_FLOAT, world_rank-1, world_rank-1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            err = MPI_Recv(buffer_right, Ny, MPI_FLOAT, world_rank+1, world_rank+1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    } else {
        if (world_rank == world_size-1) {
            err = MPI_Recv(buffer_left, Ny, MPI_FLOAT, world_rank-1, world_rank-1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            err = MPI_Send(x[0], Ny, MPI_FLOAT, world_rank-1, world_rank, MPI_COMM_WORLD);
        } else {
            // Procesadores impares no extremos
            err = MPI_Recv(buffer_left, Ny, MPI_FLOAT, world_rank-1, world_rank-1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            err = MPI_Recv(buffer_right, Ny, MPI_FLOAT, world_rank+1, world_rank+1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            err = MPI_Send(x[0], Ny, MPI_FLOAT, world_rank-1, world_rank, MPI_COMM_WORLD);
            err = MPI_Send(x[block_size-1], Ny, MPI_FLOAT, world_rank+1, world_rank, MPI_COMM_WORLD);
        }
    }

    // MATVEC LOCAL
    for (int i=0; i<block_size; i++) {
        for (int j=0; j<Ny; j++) {
            result[i][j] = C[i][j]*x[i][j];
            // donde las matrices son 0 es porque estamos en un borde
            if (N[i][j] != 0) {
                result[i][j] += N[i][j]*x[i][j+1];
            }
            if (S[i][j] != 0) {
                result[i][j] += S[i][j]*x[i][j-1];
            }
            // En W y E puede haber cambio de procesador en los bordes
            if (W[i][j] != 0) {
                if (i != 0) {
                    // estamos al interior del procesador
                    result[i][j] += W[i][j]*x[i-1][j];
                } else {
                    // estamos al borde del procesador
                    // ocupamos el resultado de la comunicacion
                    result[i][j] += W[i][j]*buffer_left[j];
                }
            }
            if (E[i][j] != 0) {
                if (i != block_size-1) {
                    // estamos al interior del procesador
                    result[i][j] += E[i][j]*x[i+1][j];
                } else {
                    // estamos al borde
                    // ocupamos resultado de comunicacion
                    result[i][j] += E[i][j]*buffer_right[j];
                }
            }
        }
    }
    free(buffer_left);
    free(buffer_right);
    return result;
}

float loc_dot_product(float** x, float** y, int n, int m) {
    // local dot product
    float result = 0.0;
    for (int i=0; i<n; i++) {
        for (int j=0; j<m; j++) {
            result += x[i][j]*y[i][j];
        }
    }
    return result;
}

float** matrix_copy(float** A, int n, int m) {
    float** result = (float**) calloc(n, sizeof(float*));
    for (int i=0; i<n; i++) {
        result[i] = (float*) calloc(m, sizeof(float));
    }
    for (int i=0; i<n; i++) {
        for (int j=0; j<m; j++) {
            result[i][j] = A[i][j];
        }
    }
    return result;
}

float dot_product(float** x, float** y, int n, int m, int world_rank, int world_size) {
    float local = loc_dot_product(x, y, n, m);
    float result = 0.0;
    int err;
    float buffer;
    if (world_rank == 0) {
        for (int i=1; i<world_size; i++) {
            err = MPI_Recv(&buffer, 1, MPI_FLOAT, i, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            result += buffer;
        }
        result += local;
        for (int i=1; i<world_size; i++) {
            err = MPI_Send(&result, 1, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
        }
    } else {
        err = MPI_Send(&local, 1, MPI_FLOAT, 0, world_rank, MPI_COMM_WORLD);
        err = MPI_Recv(&result, 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    return result;
}

float** vector_sum(float** a, float** b, int n, int m) {
    // vector sum a+b
    float** result = (float**) calloc(n, sizeof(float*));
    for (int i=0; i<n; i++) {
        result[i] = (float*) calloc(m, sizeof(float));
    }
    for (int i=0; i<n; i++) {
        for (int j=0; j<m; j++) {
            result[i][j] = a[i][j] + b[i][j];
        }
    }
    return result;
}

float** vector_scaled(float a, float** x, int n, int m) {
    float** result = (float**) calloc(n, sizeof(float*));
    for (int i=0; i<n; i++) {
        result[i] = (float*) calloc(m, sizeof(float));
    }
    for (int i=0; i<n; i++) {
        for (int j=0; j<m; j++) {
            result[i][j] = a*x[i][j];
        }
    }
    return result;
}

int main() {
    MPI_Init(NULL, NULL);
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    // Parametros
    int Nx = 100;
    int Ny = 100;
    float tol = 0.0001;
    int block_size = Nx/world_size;
    if (world_rank == world_size-1) {
        block_size = Nx/world_size + Nx%world_size;
    }

    // Creacion de matrices
    float** N = N_generator(Nx, Ny, world_rank, world_size);
    float** S = S_generator(Nx, Ny, world_rank, world_size);
    float** E = E_generator(Nx, Ny, world_rank, world_size);
    float** W = W_generator(Nx, Ny, world_rank, world_size);
    float** C = C_generator(Nx, Ny, world_rank, world_size);

    float** x = random_vector(Nx, Ny, world_rank, world_size);
    float** f = function_vector(Nx, Ny, world_rank, world_size);

    // algoritmo del libro eikjout
    float** r = vec_diff(
        mat_vec(N, S, E, W, C, x, Nx, Ny, world_rank, world_size),
        f,
        block_size,
        Ny
    );

    int i = 0;
    float error = 100000;
    float rho1, rho2, beta, delta;
    rho1 = 0.0;
    float** p;
    float** q;
    while (i < Nx*Ny && error > tol) {
        rho2 = rho1;
        rho1 = dot_product(r, r, block_size, Ny, world_rank, world_size);
        if (i==0) {
            p = matrix_copy(r, block_size, Ny);
        }
        else {
            beta = rho1/rho2;
            p = vector_sum(
                r,
                vector_scaled(beta, p, block_size, Ny),
                block_size,
                Ny
            );
        }
        q = mat_vec(N, S, E, W, C, p, Nx, Ny, world_rank, world_size);
        delta = rho1/dot_product(p, q, block_size, Ny, world_rank, world_size);
        x = vec_diff(
            x,
            vector_scaled(delta, p, block_size, Ny),
            block_size,
            Ny
        );
        r = vec_diff(
            r,
            vector_scaled(delta, q, block_size, Ny),
            block_size,
            Ny
        );
        error = rho1;
        if (i%133 == 0 && world_rank==i%world_size) {
            printf("Error: %f, procesador %i de %i con nombre %s\n", error, world_rank, world_size, processor_name);
        }
        i += 1;
    }
    
    free_matrix(q, block_size);
    free_matrix(p, block_size);
    free_matrix(r, block_size);
    free_matrix(N, block_size);
    free_matrix(S, block_size);
    free_matrix(E, block_size);
    free_matrix(W, block_size);
    free_matrix(C, block_size);
    free_matrix(x, block_size);
    free_matrix(f, block_size);
    
    MPI_Finalize();
}

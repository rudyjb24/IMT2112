#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>


// Función para rellenar los stencils.
void stencil(int i, int j, int start_i, double hx, double hy, 
            double** C, double** N, double** E, double** S, double** W) {

    double Nij = - (i * hx * (i * hx - 1) * (j + 1 / 2) * hy * ((j + 1 / 2) * hy - 1) + 1) / pow(hy, 2);
    N[i - start_i][j] = Nij;
    double Sij = - (i * hx * (i * hx - 1) * (j - 1 / 2) * hy * ((j - 1 / 2) * hy - 1) + 1) / pow(hy, 2);
    S[i - start_i][j] = Sij;

    double Eij = - ((i + 1 / 2) * hx * ((i + 1 / 2) * hx - 1) * j * hy * (j * hy - 1) + 1) / pow(hx, 2);
    E[i - start_i][j] = Eij;
    double Wij = - ((i - 1 / 2) * hx * ((i - 1 / 2) * hx - 1) * j * hy * (j * hy - 1) + 1) / pow(hx, 2);
    W[i - start_i][j] = Wij;

    double Cij = - (Wij + Eij) - (Sij + Nij) + 1;
    C[i - start_i][j] = Cij;
}

// Función para generar los stencils.
double*** stencils_generator(int start_i, int end_i, int Nx, int Ny) {
    int rows = end_i - start_i;
    double hx = double(1) / (Nx - 1);
    double hy = double(1) / (Ny - 1);

    double** C = (double**) calloc(rows, sizeof(double*));
    double** N = (double**) calloc(rows, sizeof(double*));
    double** E = (double**) calloc(rows, sizeof(double*));
    double** S = (double**) calloc(rows, sizeof(double*));
    double** W = (double**) calloc(rows, sizeof(double*));

    for (int i = 0; i < rows; i++) {
        C[i] = (double*) calloc(Nx, sizeof(double));
        N[i] = (double*) calloc(Nx, sizeof(double));
        E[i] = (double*) calloc(Nx, sizeof(double));
        S[i] = (double*) calloc(Nx, sizeof(double));
        W[i] = (double*) calloc(Nx, sizeof(double));
    }

    for (int i = start_i; i < end_i; i++) {
        for (int j = 0; j < Nx; j++) {
            stencil(i, j, start_i, hx, hy, C, N, E, S, W);
        }
    }

    double*** stencils = (double***) calloc(5, sizeof(double**));
    stencils[0] = C;
    stencils[1] = N;
    stencils[2] = E;
    stencils[3] = S;
    stencils[4] = W;

    return stencils;
}

// Función para generar arreglo 2D.
double** array_generator(int rows, int columns) {
    // Iniciamos con un arreglo de puros 0s.
    double** array = (double**) calloc(rows, sizeof(double*));

    for (int i = 0; i < rows; i++) {
        array[i] = (double*) calloc(columns, sizeof(double));
    }

    return array;
}

// Genera una función puntual, como la mencionada en el enunciado.
double** puntual_function(int rows, int columns, int rank, int size) {
    double** array = array_generator(rows, columns);

    // Haremos que solamente el primer punto interior
    // de arriba hacia abajo sea igual a 1.
    if (rank == 0) {
        array[1][1] = 1;
    }
    return array;
}

// Genera un función con valores aleatorios entre 0 y 1, en todos
// los nodos interiores.
double** random_function(int rows, int columns, int rank, int size) {
    double** array = array_generator(rows, columns);

    // Seteamos una semilla constante, para poder monitorear los resultados.
    // Todos los elementos interiores serán un número entre 0 y 1.
    for (int i = 1; i < rows - 1; i++) {
        for (int j = 1; j < columns - 1; j++) {
            array[i][j] = ((double) rand() / (RAND_MAX));
        }
    }
    return array;
}

// Suma de dos arrays.
void array_sum(double** array1, double** array2, double** result, int rows, int columns) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            result[i][j] = array1[i][j] + array2[i][j];
        }
    }
}

// Resta de dos arrays.
void array_subs(double** array1, double** array2, double** result, int rows, int columns) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            result[i][j] = array1[i][j] - array2[i][j];
        }
    }
}

// Producto punto entre dos arrays. Notemos que es el producto punto por coordenada.
double array_dot_product(double** array1, double** array2, int rows, int columns) {
    double result = 0;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            result += array1[i][j] * array2[i][j];
        }
    }
    
    return result;
}

// Multiplicación de array bidimensional por constante.
void array_by_constant(double** array, double** result, double beta, int rows, int columns) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            result[i][j] = beta * array[i][j];
        }
    }
}

// Norma al cuadrado de arreglo 2D. Notemos que es calculada por coordenada.
double vec_norm_sq(double** array, int rows, int columns) {
    double result = 0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            result += pow(array[i][j], 2);
        }
    }
    
    return result;
}

// Función para copiar un array en otra dirección de memoria.
void copy_array(double** destination, double** to_copy, int rows, int columns) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            destination[i][j] = to_copy[i][j];
        }
    }
}

// Matriz por vector, en formato stencil.
void mat_vec(double*** stencils, double** x, double* lower_neighbour,
    double* upper_neighbour, double** result, int rank, int size, int rows, int columns) {
        int start_i = 0;
        int end_i = rows;
        if (rank == 0) {
            start_i = 1;
        }
        if (rank == size - 1) {
            end_i = rows - 1;
        }
        // Iteramos sobre los puntos interiores.
        for (int i = start_i; i < end_i; i++) {
            for (int j = 1; j < columns - 1; j++) {
                double value = 0;

                // Sumamos Cij.
                value += stencils[0][i][j] * x[i][j];

                // Sumamos Nij.
                if (i == 0) {
                    value += stencils[1][i][j] * upper_neighbour[j];
                } else {
                    value += stencils[1][i][j] * x[i - 1][j];
                }

                // Sumamos Eij.
                value += stencils[2][i][j] * x[i][j + 1];

                // Sumamos Sij.
                if (i == rows - 1) {
                    value += stencils[3][i][j] * lower_neighbour[j];
                } else {
                    value += stencils[3][i][j] * x[i + 1][j];
                }

                // Sumamos Wij.
                value += stencils[4][i][j] * x[i][j - 1];

                result[i][j] = value;
            }
        }
}

// Función para liberar matrices/arreglos 2D.
void free_matrix(double** matrix, int rows) {
    for (int i = 0; i < rows; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

// Función para liberar el arreglo de stencils.
void free_stencils(double*** stencils, int rows) {
    for (int j = 0; j < 5; j++) {
        free_matrix(stencils[j], rows);
    }
    free(stencils);
}

// Función para imprimir matrices/arreglos 2D.
void print_matrix(double** matrix, int rows, int columns) {
    printf("\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            printf("%lf ", matrix[i][j]); 
        }
    printf("\n");
  }
}

// Función para comunicar las filas necesarias entre procesadores.
void vec_communication(double** x, int world_rank, int world_size, int Nx, int err, 
  int local_rows, double* lower_neighbour, double* upper_neighbour) {
    // Si el procesador es par, recibimos y luego enviamos.
    if (world_rank % 2 == 0) {
        // Si el procesador no es el último, pedimos la primera fila del siguiente.
        if (world_rank < world_size - 1) {
            err = MPI_Recv(lower_neighbour, Nx, MPI_DOUBLE, world_rank + 1, 0,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        // Si el procesador no es el primero, pedimos la última fila del anterior.
        if (world_rank > 0) {
            err = MPI_Recv(upper_neighbour, Nx, MPI_DOUBLE, world_rank - 1, 1,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // Si el procesador no es el último, enviamos la última fila al procesador siguiente.
        if (world_rank < world_size - 1) {
            err = MPI_Send(x[local_rows - 1], Nx, MPI_DOUBLE, world_rank + 1, 2, MPI_COMM_WORLD);
        }
        // Si el procesador no es el primero, enviamos la primera fila al procesador anterior.
        if (world_rank > 0) {
            err = MPI_Send(x[0], Nx, MPI_DOUBLE, world_rank - 1, 3, MPI_COMM_WORLD);
        }
    } else {
        // Si el procesador no es el primero, enviamos la primera fila al procesador anterior.
        if (world_rank > 0) {
            err = MPI_Send(x[0], Nx, MPI_DOUBLE, world_rank - 1, 0, MPI_COMM_WORLD);
        }
        // Si el procesador no es el último, enviamos la última fila al procesador siguiente.
        if (world_rank < world_size - 1) {
            err = MPI_Send(x[local_rows - 1], Nx, MPI_DOUBLE, world_rank + 1, 1, MPI_COMM_WORLD);
        }

        // Si el procesador no es el primero, pedimos la última fila del anterior.
        if (world_rank > 0) {
            err = MPI_Recv(upper_neighbour, Nx, MPI_DOUBLE, world_rank - 1, 2,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        // Si el procesador no es el último, pedimos la primera fila del siguiente.
        if (world_rank < world_size - 1) {
            err = MPI_Recv(lower_neighbour, Nx, MPI_DOUBLE, world_rank + 1, 3,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
}

// Función para el algoritmo del gradiente conjugado.
void conjugate_gradient(double*** A, double** x, double** r, double** z, double** result,
  double** q, double** p, double** b, double* lower_neighbour, double* upper_neighbour, double* remainder,
  int rank, int size, int rows, int columns, int iterations, int err, int Nx) {
    // Necesitamos los x de los vecinos.
    vec_communication(x, rank, size, Nx, err, rows, lower_neighbour, upper_neighbour);
    
    // r(0) = Ax(0) - b.
    mat_vec(A, x, lower_neighbour, upper_neighbour, result, rank, size, rows, columns);
    array_subs(result, b, r, rows, columns);
    
    double new_rho;
    double old_rho;
    double beta;
    for (int t = 0; t < iterations; t++){ 

        // Kz(t - 1) = r(t - 1).
        copy_array(z, r, rows, columns);

        // rho(t - 1) = r(t - 1)^{T}z(t - 1).
        new_rho = array_dot_product(z, r, rows, columns);

        // Sumamos todos los new_rho, y los guardamos en el mismo lugar.
        MPI_Allreduce(MPI_IN_PLACE, &new_rho, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        
        if (t == 0) {
            // p(1) = z(0).
            copy_array(p, z, rows, columns);
        } else {
            // beta(t - 1) = rho(t - 1) / rho(t - 2).
            beta = new_rho / old_rho;

            // p(t) = z(t - 1) + beta(t - 1)p(t - 1).
            array_by_constant(p, result, beta, rows, columns);
            array_sum(z, result, p, rows, columns);
        }

        // Necesitamos los p de los vecinos.
        vec_communication(p, rank, size, Nx, err, rows, lower_neighbour, upper_neighbour);
        
        // q(t) = Ap(t).
        mat_vec(A, p, lower_neighbour, upper_neighbour, q, rank, size, rows, columns);

        // delta(t) = rho(t - 1) / (p(t)^{T}q(t)).
        double dot_p_q = array_dot_product(p, q, rows, columns);

        // Sumamos todos los dot productos entre p y q, y los guardamos en el mismo lugar.
        MPI_Allreduce(MPI_IN_PLACE, &dot_p_q, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        double delta = new_rho / dot_p_q;

        // x(t) = x(t - 1) - delta(t)p(t).
        array_by_constant(p, result, delta, rows, columns);
        array_subs(x, result, x, rows, columns);

        // r(t) = r(t - 1) - delta(t)q(t).
        array_by_constant(q, result, delta, rows, columns);
        array_subs(r, result, r, rows, columns);

        // Conectamos los nuevos x.
        vec_communication(x, rank, size, Nx, err, rows, lower_neighbour, upper_neighbour);
        old_rho = new_rho;

        // Calculemos la norma de Ax - b.
        mat_vec(A, x, lower_neighbour, upper_neighbour, result, rank, size, rows, columns);
        array_subs(result, b, result, rows, columns);
        double res_norm = vec_norm_sq(result, rows, columns);

        // Sumamos todos los cuadrados de la norma del residuo local, 
        // y los guardamos en el mismo lugar.
        MPI_Allreduce(MPI_IN_PLACE, &res_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        // Si es el primer procesador.
        if (rank == 0) {
            // Imprimimos la norma del residuo.
            // printf("Iteración: %i, Norma del residuo: %lf\n", t, res_norm);
            // Guardamos el residuo en un arreglo.
            remainder[t] = res_norm;
        }
    }
}

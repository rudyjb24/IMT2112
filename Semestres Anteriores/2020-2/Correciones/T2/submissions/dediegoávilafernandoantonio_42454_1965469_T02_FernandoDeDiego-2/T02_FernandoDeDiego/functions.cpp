#include "functions.h"

// Definimos las funciones para generar matrices y vectores,
// de manera similar que en la ayudantía.
float** matrix(int n, int range) {
    // Esta será una matriz simétrica.
    // Descomentar si se quiere que la matriz sea 100% aleatoria.
    // srand((int)time(0));  
    srand(0);
    float** matrix = (float**) calloc(n, sizeof(float*));

    for (int i = 0; i < n; i++) {
        matrix[i] = (float*) calloc(n, sizeof(float));
    }
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            float value = 2 * range * ((float)rand()/(float)RAND_MAX) - range;
            matrix[i][j] = value;
            matrix[j][i] = value;
        }
    }
    return matrix;
}

float* vector(int n, int range) {
    // Descomentar si se quiere que el vector sea 100% aleatorio.
    // srand((int)time(0));
    srand(1);
    float* vector = (float*) calloc(n, sizeof(float));
    for (int i = 0; i < n; i++) {
        vector[i] = 2 * range * ((float)rand()/(float)RAND_MAX) - range;
    }
    return vector;
}

// Definimos una función para copiar un vector en otra dirección.

float* copy_vector(float* vector, int n) {
    float* copy = (float*) calloc(n, sizeof(float));
    for (int i = 0; i < n; i++) {
        copy[i] = vector[i];
    }
    return copy;
}

// Definimos una función para liberar el espacio de memoria ocupado
// por una matriz, de la misma forma que en ayudantía.
void free_matrix(float** matrix, int n) {
    for (int i = 0; i < n; i++) {
        free(matrix[i]);
    }
    free(matrix);
}


// Definimos las funciones que imprimen matrices y vectores, igual
// que en ayudantía.
void print_matrix(float** matrix, int n) {
    printf("\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%f", matrix[i][j]);
        }
        printf("\n");
    }
}

void print_vector(float* vector, int n) {
    printf("\n");
    for (int i = 0; i < n; i++) {
        printf("%f\n", vector[i]);
    }
}

// Definimos una función que calcula la norma 2 de un vector.

// Versión lineal.
float vec_norm(float* vector, int n) {
    float sum = 0;
    for (int i = 0; i < n; i++) {
        sum += pow(vector[i], 2);
    }
    return sqrt(sum);
}

// Versión paralelizada.
float par_vec_norm(float* vector, int n, int t) {
    float sum = 0;
    #pragma omp parallel for num_threads(t)
    for (int i = 0; i < n; i++) {
        sum += pow(vector[i], 2);
    }
    return sqrt(sum);
}

// Definimos una función que multiplica matriz por vector,
// de la misma forma que en la ayudantía.

// Versión lineal.
void mat_vec(float* Ab, float** A, float* b, int n) {
    for (int i = 0; i < n; i++)  {
        Ab[i] = 0;
        for (int j = 0; j < n; j++) {
            Ab[i] += A[i][j] * b[j];
        }
    }
}

// Versión paralelizada.
void par_mat_vec(float* Ab, float** A, float* b, int n, int t) {
    #pragma omp parallel for num_threads(t)
    for (int i = 0; i < n; i++)  {
        Ab[i] = 0;
        for (int j = 0; j < n; j++) {
            Ab[i] += A[i][j] * b[j];
        }
    }
}

// Definimos una función que calcula el producto interno
// entre dos vectores.

// Versión lineal.
float inner_product(float* vec1, float* vec2, int n) {
    float sum = 0;
    for (int i = 0; i < n; i++) {
        sum += vec1[i] * vec2[i];
    }
    return sum;
}

// Versión paralelizada.
float par_inner_product(float* vec1, float* vec2, int n, int t) {
    float sum = 0;
    // Nota: reduction(+:sum) hace que cada thread calcule
    // privadamente una porción del resultado, y luego
    // la acumula en sum.
    #pragma omp parallel for reduction(+:sum) num_threads(t)
    for (int i = 0; i < n; i++) {
        sum += vec1[i] * vec2[i];
    }
    return sum;
}

// Definimos una función que calcule el siguiente b.

// Versión lineal.
void b_iteration(float* Ab, float* b, int n) {
    float norm = vec_norm(Ab, n);
    for (int i = 0; i < n; i++) {
        b[i] = Ab[i] / norm;
    }
}

// Versión paralelizada.
void par_b_iteration(float* Ab, float* b, int n, int t) {
    float norm = par_vec_norm(Ab, n, t);
    for (int i = 0; i < n; i++) {
        b[i] = Ab[i] / norm;
    }
}

// Definimos una función que calcule el valor de la
// secuencia que converge al valor propio más grande.

// Versión lineal.
float mu_iteration(float* Ab, float* b, int n) {
    float num = inner_product(b, Ab, n);
    float den = inner_product(b, b, n);
    return num / den;
}

// Versión paralelizada.
float par_mu_iteration(float* Ab, float* b, int n, int t) {
    float num = par_inner_product(b, Ab, n, t);
    float den = par_inner_product(b, b, n, t);
    return num / den;
}

// Definimos una función para correr el método de potencia.

// Versión lineal.
void power_iteration(float** A, float* b, float* mu, int n, int iter) {
    float* Ab = (float*) calloc(n, sizeof(float));
    for (int k = 0; k < iter; k++) {
        mat_vec(Ab, A, b, n);  // Multiplicación de A y b.
        mu[k] = mu_iteration(Ab, b, n);  // Valor propio.
        b_iteration(Ab, b, n);  // Actualización de b.
    }
    free(Ab);
}

// Versión paralelizada.
void par_power_iteration(float** A, float* b, float* mu, int n, int iter, int t) {
    float* Ab = (float*) calloc(n, sizeof(float));
    for (int k = 0; k < iter; k++) {
        par_mat_vec(Ab, A, b, n, t);  // Multiplicación de A y b.
        mu[k] = par_mu_iteration(Ab, b, n, t);  // Valor propio.
        par_b_iteration(Ab, b, n, t);  // Actualización de b.
    }
    free(Ab);
}

// Definimos una función que corra el algoritmo, y calcule
// el tiempo de ejecución.
void run_algorithm(float** A, float* b, float* mu, int n, int iter, int t) {
    auto start = std::chrono::high_resolution_clock::now();
    // Si t = 0, corremos la versión lineal.
    if (t == 0) {
        power_iteration(A, b, mu, n, iter);
    } else if (t > 0) {
        par_power_iteration(A, b, mu, n, iter, t);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto execution = std::chrono::duration_cast<std::chrono::nanoseconds>( end - start); 
    printf("Time measured: %.3f seconds.\n", execution.count() * 1e-9);
}

// Definimos una función que corra el algoritmo, y calcule
// el tiempo de ejecución.
float return_time(float** A, float* b, float* mu, int n, int iter, int t) {
    auto start = std::chrono::high_resolution_clock::now();
    // Si t = 0, corremos la versión lineal.
    if (t == 0) {
        power_iteration(A, b, mu, n, iter);
    } else if (t > 0) {
        par_power_iteration(A, b, mu, n, iter, t);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto execution = std::chrono::duration_cast<std::chrono::nanoseconds>( end - start);

    return execution.count() * 1e-9;
}

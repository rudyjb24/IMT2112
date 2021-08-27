#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <chrono>
#include <math.h>

float* vector_generator(int n, int range) {
    // generates n-dimensional vector
    srand((int)time(0));
    float* x = (float*) calloc(n, sizeof(float));
    if (x == NULL) {
        printf("Memory not allocated");
    }
    for (int i=0; i<n; i++) {
        x[i] = 2*range*((float)rand()/(float)RAND_MAX) - range;
    }
    return x;
}

float** matrix_generator(int n, int m, int range) {
    // generates n x m matrix
    srand((int)time(0));
    float** matrix = (float**) calloc(n, sizeof(float*));
    if (matrix == NULL) {
        printf("memory not allocated");
    }
    for (int i=0; i<n; i++) {
        matrix[i] = (float*) calloc(m, sizeof(float));
        if (matrix[i] == NULL) {
            printf("memory not allocated");
        }
    }
    for (int i=0; i<n; i++) {
        for (int j=0; j<m; j++) {
            matrix[i][j] = 2*range*((float)rand()/(float)RAND_MAX) - range;
        }
    }
    // to improve chances of convergence due to eigenvalue distribution
    // given by Gershgorin theorem
    matrix[0][0] += 100*range;
    return matrix;
}

void print_matrix(float** matrix, int n, int m) {
    printf("\n");
    for (int i=0; i<n; i++) {
        for (int j=0; j<m; j++) {
            printf("%f ", matrix[i][j]);
        }
        printf("\n");
    }
}

void print_vector(float* vector, int n) {
    printf("\n[");
    for (int i=0; i<n; i++) {
        if (i < n-1) {
            printf("%f ", vector[i]);
        }
        else {
            printf("%f", vector[i]);
        }
    }
    printf("]\n");
}

void free_matrix(float** matrix, int n) {
    for (int i=0; i<n; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

float* parallel_matvec(float** A, float* x, int n, int m, int n_threads) {
    // parecido al código de la ayudantía 4
    float* y = (float*) calloc(n, sizeof(float));
    if (y == NULL) {
        printf("memory not allocated");
    }
    #pragma omp parallel for num_threads(n_threads)
    for (int i=0; i<n; i++) {
        for (int j=0; j<m; j++) {
            y[i] += A[i][j]*x[j];
        }
    }
    return y;
}

float* matvec(float** A, float* x, int n, int m) {
    float* y = (float*) calloc(n, sizeof(float));
    if (y == NULL) {
        printf("memory not allocated");
    }
    for (int i=0; i<n; i++) {
        for (int j=0; j<m; j++) {
        y[i] += A[i][j]*x[j];
        }
    }
    return y;
}

float norm(float* x, int n) {
    // returns 2-norm of vector x
    float result = 0.0;
    for (int i=0; i<n; i++) {
        result += x[i]*x[i];
    }
    return sqrt(result);
}

float parallel_norm(float* x, int n, int n_threads) {
    // parallel 2-norm of vector x with n_threads number of threads
    float sum = 0.0;
    float* thread_sum = (float*) calloc(n_threads, sizeof(float));
    #pragma omp parallel for num_threads(n_threads)
    for (int i=0; i<n; i++) {
        int id = omp_get_thread_num();
        // threadsafe sum
        thread_sum[id] += x[i]*x[i];
    }
    for (int i=0; i<n_threads; i++) {
        sum += thread_sum[i];
    }
    return sqrt(sum);
}

float dot(float* x, float* y, int n) {
    float dot = 0.0;
    for (int i=0; i<n; i++) {
        dot += x[i]*y[i];
    }
    return dot;
}

float parallel_dot(float* x, float* y, int n, int n_threads) {
    // parallel dot-product with n_threads number of threads
    float sum = 0.0;
    float* thread_sum = (float*) calloc(n_threads, sizeof(float));
    #pragma omp parallel for num_threads(n_threads)
    for (int i=0; i<n; i++) {
        int id = omp_get_thread_num();
        // threadsafe sum
        thread_sum[id] += x[i]*y[i];
    }
    for (int i=0; i<n_threads; i++) {
        sum += thread_sum[i];
    }
    return sum;
}

float parallel_power_method(float** A, int n, int max_iter, int n_threads) {
    // returns highest eigenvalue approximation of square matrix A
    // after max_iter iterations
    // random initial vector
    float* x0 = vector_generator(n, 100);
    float mu;
    float x0_norm;
    for (int i=0; i<max_iter; i++) {
        x0 = parallel_matvec(A, x0, n, n, n_threads);
        // vector scaling
        x0_norm = parallel_norm(x0, n, n_threads);
        #pragma omp parallel for num_threads(n_threads)
        for (int j=0; j<n; j++) {
            x0[j] = x0[j]/x0_norm;
        }
        mu = parallel_dot(x0, parallel_matvec(A, x0, n, n, n_threads), n, n_threads);
    }
    free(x0);
    return mu;
}

float power_method(float** A, int n, int max_iter) {
    // returns highest eigenvalue approximation of square matrix A
    // after max_iter iterations
    // random initial vector
    float* x0 = vector_generator(n, 100);
    float mu;
    float x0_norm;
    for (int i=0; i<max_iter; i++) {
        x0 = matvec(A, x0, n, n);
        x0_norm = norm(x0, n);
        // vector scaling
        for (int j=0; j<n; j++) {
            x0[j] = x0[j]/x0_norm;
        }
        mu = dot(x0, matvec(A, x0, n, n), n);
    }
    free(x0);
    return mu;
}

float* parallel_vecdiff(float* x, float*y, int n, int n_threads) {
    // only for testing purposes
    float* z = (float*) calloc(n, sizeof(float));
    if (z == NULL) {
        printf("memory not allocated");
    }
    #pragma omp parallel for num_threads(n_threads)
    for (int i=0; i<n; i++) {
        z[i] = x[i]-y[i];
    }
    return z;
}

int main() {

    int thr[4] = {2, 4, 8, 12};

    printf("\nTAREA 2 \n");
    printf("Vicente Hojas \n\n");

    for(int exp=2; exp<3; exp++) {
        // k is thread array index
        for (int k=3; k<4; k++) {
            int n = (int) pow((double) 10, (double) exp);
            float** A = matrix_generator(n, n, 100.0);
            int n_threads = thr[k];
            // NUMBER OF ITERATIONS
            int max_iter = 100;

            printf("-------------------------------------------------------\n");
            printf("number of threads = %i\n", n_threads);
            printf("size of matrix = %i x %i\n\n", n, n);

            auto start = std::chrono::high_resolution_clock::now();
            float mu1 = power_method(A, n, max_iter);
            auto end = std::chrono::high_resolution_clock::now();
            auto execution = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start);
            printf("Time measured (seq): %.3f seconds.\n", execution.count() * 1e-9);

            auto start_p = std::chrono::high_resolution_clock::now();
            float mu2 = parallel_power_method(A, n, max_iter, n_threads);
            auto end_p = std::chrono::high_resolution_clock::now();
            auto execution_p = std::chrono::duration_cast<std::chrono::nanoseconds>(end_p-start_p);
            printf("Time measured (par): %.3f seconds.\n", execution_p.count()* 1e-9);

            printf("mu (seq) = %f\n", mu1);
            printf("mu (par) = %f\n", mu2);
            printf("relative error = %f\n", abs(mu1-mu2)/abs(mu1));

            free_matrix(A, n);

        }
    }

    // norm testing
    // float* x_test = (float*) calloc(10000, sizeof(float));
    // for (int i=0; i<10000; i++) {
    //     x_test[i] = 1.0;
    // }
    // float* x = vector_generator(10000, 10.0);

    // printf("norm (seq) of x_test = %f\n", norm(x_test, 10000));
    // printf("norm (par) of x_test = %f\n", parallel_norm(x_test, 10000, 12));

    // printf("norm (seq) of random x = %f\n", norm(x, 10000));
    // printf("norm(par) of random x = %f\n", parallel_norm(x, 10000, 12));

    // free(x_test);
    // free(x);
}

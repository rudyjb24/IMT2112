#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <iostream>

float* vector_generator(int n, int range) {
  srand((int)time(0));
  float* vector = (float*) calloc(n, sizeof(float));
  
  for (int i=0; i<n; i++) {
    vector[i] = (2*range * ((float)rand() / (float)RAND_MAX)) - range;
  }

  return vector;
}

void print_vector(float* vector, int n) {
  printf("\n");
  for (int i=0; i<n; i++) {
      printf("%f\n", vector[i]);
  }
}

float** matrix_generator(int n, int range) {
  srand((int)time(0));
  float** matrix = (float**) calloc(n, sizeof(float*));

  for (int i = 0; i < n; i++) {
    matrix[i] = (float*) calloc(n, sizeof(float));
    // matrix[i] = vector_generator(n, range);
  }

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (j >= i) {
        matrix[i][j] = (2*range * ((float)rand() / (float)RAND_MAX)) - range;
      }
    }
  }

  return matrix;
}

void free_matrix(float** matrix, int n) {
  for (int i = 0; i < n; i++) {
    free(matrix[i]);
  }
  free(matrix);
}

void print_matrix(float** matrix, int n) {
  printf("\n");
  for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
         printf("%f ", matrix[i][j]); 
      }
    printf("\n");
  }
}

float* mat_vec(float** matrix, float* vector, int n) {
    float* result = (float*)calloc(n, sizeof(float));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            result[i] += matrix[i][j] * vector[j];
        }
    }
  return result;
}

float* mat_vec_par(float **matrix, float *vector, int n) {
    float *result = (float*)calloc(n, sizeof(float));
    #pragma omp parallel for num_threads(4)
    for (int i = 0; i < n; i++) {
        #pragma omp parallel for num_threads(4)
        for (int j = 0; j < n; j++) {
            result[i] += matrix[i][j] * vector[j];
        }
    }
  return result;
}

int main() {
  
  int n = 50000;
  
  float* x = vector_generator(n, 100);
  float** A = matrix_generator(n, 100);

  //print_matrix(A, n);
  //print_vector(x, n);


  auto start = std::chrono::high_resolution_clock::now();
  //float* b = mat_vec(A, x, n);
  auto end = std::chrono::high_resolution_clock::now();
  auto execution = std::chrono::duration_cast<std::chrono::nanoseconds>( end - start ); 
  //print_vector(b, n);
  //printf("Time measured: %.3f seconds\n", execution.count() * 1e-9);


  auto start_p = std::chrono::high_resolution_clock::now();
  float* bp = mat_vec_par(A, x, n);
  auto end_p = std::chrono::high_resolution_clock::now();
  auto execution_p = std::chrono::duration_cast<std::chrono::nanoseconds>( end_p - start_p );
  //print_vector(bp, n);
  printf("Time measured: %.3f seconds\n", execution_p.count()* 1e-9);

  free_matrix(A, n);
  free(x);
  //free(b);
  free(bp);
  return 0;
}
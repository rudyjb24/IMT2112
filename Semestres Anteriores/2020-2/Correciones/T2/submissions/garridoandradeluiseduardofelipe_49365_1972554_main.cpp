#include <stdio.h> // print() 
#include <stdlib.h> // rand()
#include <ctime> // time()
#include <omp.h> // omp_get_thread_num()
#include <chrono> // chrono()
#include <cmath> // sqrt(), pow()


// En primer lugar, vamos a definir una función que genere
// una matriz simétrica y definida positiva nxn. De esta forma,
// vamos a obtener una matriz diagonalizable de valores propios 
// reales y positivos. 

// Para realizar lo anterior, vamos a definir las siguientes 
// operaciones básicas de matrices:

float** scale_matrix(float** matrix, float lambda, int n) {
    // Definimos el formato de una matriz n x n
    float** matrix2 = (float**) malloc(n * sizeof(float*));
    for (int i = 0; i < n; i++) {
        matrix2[i] = (float *) malloc(n * sizeof(float));
    }
    // Multiplica cada componente (matriz n x n) por lambda y se guarda
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrix2[i][j] = matrix[i][j]*lambda;
        }
    }
    return matrix2;
}

float** sum_matrix(float** matrix1, float** matrix2, int n) {
    // Definimos el formato de una matriz n x n
    float** matrix3 = (float**) malloc(n * sizeof(float*));
    for (int i = 0; i < n; i++) {
        matrix3[i] = (float *) malloc(n * sizeof(float));
    }
    // Suma cada compenente de las matrices
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrix3[i][j] = matrix1[i][j] + matrix2[i][j];
        }
    }
    return matrix3;
}

float** matrix_transpose(float** matrix, int n) {
    // Definimos el formato de una matriz n x n
    float** matrix2 = (float**) malloc(n * sizeof(float*));
    for (int i = 0; i < n; i++) {
        matrix2[i] = (float *) malloc(n * sizeof(float));
    }
    // Recorremos cada componente y lo vamos intercambiando
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrix2[i][j] = matrix[j][i];
        }
    }
    return matrix2;
}

float** identity_matrix(int n) {
    // Definimos el formato de una matriz n x n
    float** matrix = (float**) calloc(n, sizeof(float*));
    for (int i = 0; i < n; i++) {
        matrix[i] = (float *) calloc(n, sizeof(float));
    }
    // Recorremos cada componente y la diagonal se iguala a 1
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) {
            matrix[i][j] = 1;
            }
        }
    }
    return matrix;
}

// Las siguientes funciones son útiles para el uso de la
// memoria junto con la visualización de matrices y vectores.

void print_matrix(float** matrix, int n) {
    // Imprime en consola la matriz
    printf("\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%f ", matrix[i][j]); 
        }
        printf("\n");
    }
}

void print_vector(float* vector, int n) {
    // Imprime en consola el vector
    printf("\n");
    for (int i = 0; i < n; i++) {
        printf("%f ", vector[i]);
    }
    printf("\n");
}

void free_matrix(float** matrix, int n) {
  for (int i = 0; i < n; i++) {
    free(matrix[i]);
  }
  free(matrix);
}

// Ahora, definimos los generadores de matrices y vectores
// que usaremos como testeo para el algoritmo

float** matrix_generator(int n, int range) {
    // Definimos el formato de la matriz n x n
    float** matrix = (float**) calloc(n, sizeof(float*));
    for (int i = 0; i < n; i++) {
        matrix[i] = (float*) calloc(n, sizeof(float));
    }
    // Ahora, se asigna un valor pseudo-random a cada componente
    srand((int)time(0));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrix[i][j] = 2*range*((float)rand()/(float)RAND_MAX) - range;
        }
    }
    return matrix;
}

float** spd_matrix_generator(int n, int range) {
    // Generamos una matriz pseudo-random
    float** A = matrix_generator(n, range);
    // Para toda matriz A (nxn) se satisface que la matriz
    // dada por B = 0.5(A + A^T) + n*I define una matriz
    // simétrica definida positiva. De esta forma, usamos las 
    // operaciones anteriores para definir tal matriz.
    float** I = identity_matrix(n);
    float lambda1 = 0.5;
    float lambda2 = n;
    // Calculamos
    float** aux1 = matrix_transpose(A, n);
    float** aux2 = sum_matrix(A, aux1, n);
    float** aux3 = scale_matrix(aux2, lambda1, n);
    float** aux4 = scale_matrix(I, lambda2, n);
    float** B = sum_matrix(aux3, aux4, n);
    // Liberamos memoria
    free_matrix(I, n);
    free_matrix(A, n);
    // Retornamos la matriz resultante
    return B;
}

float* vector_generator(int n, int range) {
    // Definimos el formato de un vector de dimensión n x 1
    float* vector = (float*) calloc(n, sizeof(float));
    // Ahora, se asigna un valor pseudo-random a cada componente
    srand((int)time(0));
    for (int i = 0; i < n; i++) {
        vector[i] = 2*range*((float)rand()/(float)RAND_MAX) - range;
    }
    return vector;
}

// Ahora, nos podemos enfocar en el método que queremos paralelizar. 
// En este caso, podemos observar que las operaciones involucradas son
// producto punto, producto matriz-vector, norma y multiplicación
// por escalar (la división). A continuación, vamos a definir estas
// funciones en versión normal y paralelizada con OpenMP.

float inner_product(float* a, float* b, int n) {
    // Se desea retornar el producto a^T b. 
    float result = 0.0;
    // Multiplicamos cada componente respectivo y lo vamos sumando
    for (int i = 0; i < n; i++) {
        result += a[i]*b[i];
    }
    // Se retorna el resultado obtenido
    return result;
}

float inner_product_parallel(float* a, float* b, int n, int threads) {
    // Se desea retornar el producto a^T b paralelizado (thread safe)
    float result = 0.0;
    // Debemos crear un vector para guardar los datos locales de cada thread
    float* array = (float*) calloc(threads, sizeof(float));
    // Se reparte el trabajo a los procesadores
    {
        #pragma omp parallel for num_threads(threads)
        for (int i=0;i<n;i++) {
            int id = omp_get_thread_num();
            array[id] += a[i]*b[i];
        }
    }
    // Sumamos los resultados obtenidos por cada thread
    for (int i=0;i<threads;i++) {
        result += array[i]; 
        }
    // Liberamos memoria
    free(array);
    // Retornamos el resultado
    return result;
}

float* mat_vec(float** A, float* b, int n) {
    // Se desea retornar el producto A b. 
    float* result = (float *) calloc(n, sizeof(float));
    // Recorremos cada fila y multiplicamos cada componente respectivo,
    // de manera que el resultado se guarde en el array
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            result[i] += A[i][j]*b[j];
        }
    }
    // Se retorna el resultado obtenido
    return result;
}
       
float* mat_vec_parallel(float **A, float *b, int n, int threads) {   
    // Se desea retornar el producto A b.
    float *result = (float *)calloc(n, sizeof(float));
    #pragma omp parallel for num_threads(threads)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            result[i] += A[i][j] * b[j];
        }
    }
  return result;
}

float euclidian_norm(float* a, int n) {
    // Se desea retornar el producto ||a||. 
    float result = 0.0;
    // Multiplicamos cada componente respectiva por si misma
    // y lo vamos sumando al resultado
    for (int i = 0; i < n; i++) {
        result += a[i]*a[i];
    }
    // Calculamos la raíz
    result = sqrt(result);
    // Se retorna el resultado obtenido
    return result;
}

float euclidian_norm_parallel(float* a, int n, int threads) {
    // Se desea retornar ||a|| paralelizado (thread safe)
    float result = 0.0;
    // Debemos crear un vector para guardar los datos locales de cada thread
    float* array = (float*) calloc(threads, sizeof(float));
    // Se reparte el trabajo a los procesadores
    {
        #pragma omp parallel for num_threads(threads)
        for (int i=0;i<n;i++) {
            int id = omp_get_thread_num();
            array[id] += a[i]*a[i];
        }
    }
    // Sumamos los resultados obtenidos por cada thread
    for (int i=0;i<threads;i++) {
        result += array[i]; 
        }
    // Liberamos memoria
    free(array);
    // Calculamos la raiz y retornamos el resultado
    result = sqrt(result);
    return result;
}

float* scale_vector(float* a, float lambda, int n) {
    // Definimos el formato de del vector n x 1
    float* vector = (float*) malloc(n * sizeof(float));
    // Multiplica cada componente (matriz n x n) por lambda y se guarda
    for (int i = 0; i < n; i++) {
        vector[i] = a[i]*lambda;
    }
    return vector;
} 

// Ya podemos definir el algoritmo de power iteration para 
// calcular los valores propios!

float power_iteration(float** A, float* b0, int n, int max_iter) {
    // En primer lugar, definimos el vector que definirá al vector propio
    // obtenido al terminar el proceso iterativo.
    float* bk = b0;
    float* aux1 = (float *) malloc(n * sizeof(float));
    float aux2 = 0.0;
    float escalar = 0.0;
    // Definimos un contador de las iteraciones e iniciamos el loop
    int iter = 0;
    while (iter != max_iter) {
        // Usamos las funciones anteriores para calcular el vecto b_{k+1}
        aux1 = mat_vec(A, bk, n);
        aux2 = euclidian_norm(aux1, n);
        // Actualizamos de esta forma el vector bk
        escalar = 1/aux2;
        bk = scale_vector(aux1, escalar, n);
        // Aumentamos el contador
        iter += 1;
    }
    // Una vez terminado el proceso anterior, tenemos que calcular el valor propio
    float* aux3 = mat_vec(A, bk, n);
    float aux4 = inner_product(bk, aux3, n);
    float aux5 = inner_product(bk, bk, n);
    // Calculamos la razón, liberamos memoria y retornamos
    float valor_propio = aux4/aux5;
    free(aux3);
    free(bk);
    return valor_propio;
}

float power_iteration_parallel(float** A, float* b0, int n, int max_iter, int threads) {
    // En primer lugar, definimos el vector que definirá al vector propio
    // obtenido al terminar el proceso iterativo.
    float* bk = b0;
    float* aux1 = (float *) malloc(n * sizeof(float));
    float aux2 = 0.0;
    float escalar = 0.0;
    // Definimos un contador de las iteraciones e iniciamos el loop
    int iter = 0;
    while (iter != max_iter) {
        // Usamos las funciones anteriores para calcular el vecto b_{k+1}
        aux1 = mat_vec_parallel(A, bk, n, threads);
        aux2 = euclidian_norm_parallel(aux1, n, threads);
        // Actualizamos de esta forma el vector bk
        escalar = 1/aux2;
        bk = scale_vector(aux1, escalar, n);
        // Aumentamos el contador
        iter += 1;
    }
    // Una vez terminado el proceso anterior, tenemos que calcular el valor propio
    float* aux3 = mat_vec_parallel(A, bk, n, threads);
    float aux4 = inner_product_parallel(bk, aux3, n, threads);
    float aux5 = inner_product_parallel(bk, bk, n, threads);
    // Calculamos la razón, liberamos memoria y retornamos
    float valor_propio = aux4/aux5;
    free(aux3);
    free(bk);
    return valor_propio;
}

void convergence_test(float** A, float* b0, int n, int max_iter, int threads) {
    printf("--------------------------------------------------------------------\n");
    printf("               Test de Convergencia Power Iteration\n");
    printf("--------------------------------------------------------------------\n\n");
 
    printf("Se considera la siguiente matriz simétrica y definida positiva\n");
    printf("de dimensión (%i x %i): \n\n", n, n);
    print_matrix(A, n);
    printf("\n");

    printf("La convergencia del algoritmo Power Iteration se muestra a\n");
    printf("continuación: \n\n");

    float resultado;
    for (int k = 1; k <= max_iter; k++) {
        resultado = power_iteration(A, b0, n, k);
        printf("Máximo valor propio obtenido: %f ", resultado);
        printf("(Iteraciones = %i)\n", k);
    }
    printf("\n");

    printf("La convergencia del algoritmo Power Iteration (OpenMP)\n");
    printf("se muestra a continuación (%i threads): \n\n", threads);

    for (int k = 1; k <= max_iter; k++) {
        resultado = power_iteration_parallel(A, b0, n, k, threads);
        printf("Máximo valor propio obtenido: %f ", resultado);
        printf("(Iteraciones = %i)\n", k);
    }
    printf("\n");
}

void time_test(int range, int max_iter, int threads, int max_power) {
    printf("--------------------------------------------------------------------\n");
    printf("               Test de Rendimiento Power Iteration\n");
    printf("--------------------------------------------------------------------\n\n");
 
    printf("Se ejecutan ambos algoritmos para %i iteraciones!\n\n", max_iter);

    int n;
    float resultado1;
    float resultado2;

    for (int k = 1; k <= max_power; k++) {
        n = pow(10, k);
        // Definimos la matriz y vector generadas
        float** A = spd_matrix_generator(n, range);
        float* b = vector_generator(n, range);
        // Calculamos los resultados y su tiempo de ejecución
        auto start1 = std::chrono::high_resolution_clock::now();
        resultado1 = power_iteration(A, b, n, max_iter);
        auto end1 = std::chrono::high_resolution_clock::now();
        auto execution1 = std::chrono::duration_cast<std::chrono::nanoseconds>(end1 - start1);

        auto start2 = std::chrono::high_resolution_clock::now();
        resultado2 = power_iteration_parallel(A, b, n, max_iter, threads);
        auto end2 = std::chrono::high_resolution_clock::now();
        auto execution2 = std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start2);
        printf("--> Se considera una matriz de dimensión (%i x %i):\n", n, n);
        printf("Máximo valor propio obtenido: %f en %.3f segundos.\n", resultado1, 
            execution1.count() * 1e-9);
        printf("Máximo valor propio obtenido (OpenMP): %f en %.3f segundos.\n", 
            resultado2, execution2.count() * 1e-9);
        printf("\n");
        free(b);
        free_matrix(A, n);
    }
}

void threads_time_test(int n, int range, int max_iter, int max_threads) {
    printf("--------------------------------------------------------------------\n");
    printf("             Test de Threads Power Iteration (OpenMP) \n");
    printf("--------------------------------------------------------------------\n\n");
 
    printf("Se considera una matriz (%i x %i) y %i iteraciones!\n\n", n, n, max_iter);

    // Definimos la matriz y vector generadas
    float** A = spd_matrix_generator(n, range);
    float* b = vector_generator(n, range);
    // Definimos las variables de resultados
    float resultado;

    for (int k = 1; k <= max_threads; k++) {
        // Calculamos el resultado y su tiempo de ejecución
        auto start = std::chrono::high_resolution_clock::now();
        resultado = power_iteration_parallel(A, b, n, max_iter, k);
        auto end = std::chrono::high_resolution_clock::now();
        auto execution = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        printf("--> Se ejecuta el algoritmo con %i threads:\n", k);
        printf("Máximo valor propio obtenido (OpenMP): %f en %.3f segundos.\n", 
            resultado, execution.count() * 1e-9);
        printf("\n");
    }
    free(b);
    free_matrix(A, n);
}

// Finalmente, se realizará el testeo en el código principal.

int main() {
    ////////////////////////////////////////////////

    // Atributos de testeo de convergencia

    int n = 100;
    int range = 10;
    int threads = 4;
    int max_iter = 100;

    // Generamos la matriz spd y un vector cualquiera

    float** A = spd_matrix_generator(n, range);
    float* b = vector_generator(n, range);

    convergence_test(A, b, n, max_iter, threads);

    //////////////////////////////////////////////////

    // Atributos de testeo de tiempo de cómputo

    threads = 4;
    max_iter = 10;
    range = 10;
    int max_power = 4;

    // Generamos la matriz spd y un vector cualquiera

    time_test(range, max_iter, threads, max_power);

    //////////////////////////////////////////////////

    // Atributos de testeo de tiempo de cómputo

    int max_threads = 4;
    max_iter = 40;
    range = 100;
    n = 5000;

    // Generamos la matriz spd y un vector cualquiera

    threads_time_test(n, range, max_iter, max_threads);

    free(b);
    free_matrix(A, n);

    // En base a los tres test propuestos, podemos ver que el 
    // algoritmo de Power Iteration aumenta bastante su rendimiento
    // con OpenMP para dimensiones bastante grandes. Por otro lado,
    // los resultados de valores propios son bastante coherentes por
    // como se había definido la matriz simétrica y definida positiva.
    // Además, los resultantos dados por OpenMP son bastante confiables
    // ya que coincidian con los de la versión normal.

    return 0;
}

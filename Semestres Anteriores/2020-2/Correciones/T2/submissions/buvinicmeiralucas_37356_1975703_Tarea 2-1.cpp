#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <omp.h>
#include <cmath>
#include <chrono>
#include <iostream>

//Ocupamos double en vez de float para mayor precisión.
double** matrix_generator(int n, int range) //1. Función que crea una matriz aleatoria con valores en el rango "range" y de tamaño nxn. Adaptado del código de mat_vec.cpp de ayudantías.
{
    srand((int) time(0));
    double** matrix = (double**) calloc(n, sizeof(double*)); //2. b. Ocupamos calloc para guardar memoria en el Header y así poder hacer dynamic memory allocation.
    for(int i = 0; i < n; i++)
    {
        matrix[i] = (double*) calloc(n, sizeof(double));
    }
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < n; j++)
        {
            matrix[i][j] = 2*range*((double) rand()/(double) RAND_MAX) - range;
        }
    }
    return matrix;
}

void free_matrix(double** matrix, int n) //Libera la memoria en el Header ocupada por la matriz creada aleatoriamente. Adaptado del código de mat_vec.cpp de ayudantías.
{
    for(int i = 0; i < n; i++)
    {
        free(matrix[i]);
    }
    free(matrix);
}

double* vector_generator(int n, int range) //Crea un vector aleatorio de tamaño n y con valores en el rango "range". Adaptado del código de mat_vec.cpp de ayudantías.
{
    srand((int) time(0));
    double* vec = (double*) calloc(n, sizeof(double)); //Hacemos también dynamic memory allocation con calloc.
    for(int i = 0; i < n; i++)
    {
        vec[i] = 2*range*((double) rand()/(double) RAND_MAX) - range;
    }
    return vec;
}

void print_arr(double* arr, int n) //Imprime un array de tamaño nx1.
{
    printf("\n");
    for(int i = 0; i < n; i++)
    {
        printf("%f \n", arr[i]);
    }
}

double* mat_vec(double** matrix, double* vec, int n, int thre) //2. c. Multiplica la matriz nxn y el vector nx1 aleatoriamente generados. Adaptado del código de mat_vec.cpp de ayudantías.
{
    double* result = (double*) calloc(n, sizeof(double)); //Ocupamos también dynamic memory allocation.
    #pragma omp parallel for num_threads(thre) //Aplicamos paralelismo con thre cantidad de hilos. Esta implementación es thread safe, ya que ocupamos el comando "for" que produce que la tarea sea dividida en los "thre" hilos y también porque el objeto que es modificado es un array y los hilos siempre modifican entradas distitnas de este, evitando que pueda suceder un race condition.
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < n; j++)
        {
            result[i] += matrix[i][j] * vec[j];
        }
    }
    return result;
}

double dot_product(double* vec1, double* vec2, int n, int thre) //Calcula el producto punto entre dos vectores (y, por consecuencia, agregandole un sqrt de cmath, calcula la norma) con paralelismo.
{
    double v = 0.0;
    double* arr = (double*) calloc(thre, sizeof(double)); //Para evitar race condition creamos un array "arr" de tamaño "thre" en que pueden sumar los "thre" hilos.
    #pragma omp parallel for num_threads(thre) //Como también ocupamos el comando "for" para el paralelismo, este es thread safe.
    for(int i = 0; i < n; i++)
    {
        int id = omp_get_thread_num();
        arr[id] += vec1[i] * vec2[i];
        //v += vec1[i] * vec2[i];
    }
    for (int i = 0; i < thre; i++)
    {
        v += arr[i];
    }
    free(arr);
    return v;
}

double* vec_scal(double* vec, double c, int n, int thre) //Entrega la multiplicación entre un vector y un escalar "c".
{
    double* result = (double*) calloc(n, sizeof(double));
    #pragma omp parallel for num_threads(thre) //También ocupamos paralelismo thread safe (análogamente a como lo implementamos en la función mat_vec).
    for(int i = 0; i < n; i++)
    {
        result[i] = c * vec[i];
    }
    return result;
}

double* power_iteration(double** A, double* bk, int n, int thre) //Crea una matriz y vector inicial aleatorios (tamaños nxn y nx1) y entrega una iteración del método de la potencia hecho con "thre" hilos.
{
    double* Ab = mat_vec(A, bk, n, thre);
    double normAb = sqrt(dot_product(Ab, Ab, n, thre));
    double* result = vec_scal(Ab, 1/normAb, n, thre);
    free(Ab);
    return result;
}

double dominant_eig(double** A, double* bk, int n, int thre) //Calcula el eigenvalor dominante.
{
    double* Ab = mat_vec(A, bk, n, thre);
    double uk = dot_product(Ab, bk, n, thre) / dot_product(bk, bk, n, thre);
    free(Ab);
    return uk;
}

double* thousand_eigs(int n, int thre) //2. d. Entrega un array con los útlimos 100 eigenvalores resultantes de las 10000 iteraciones del método de la potencia. Esta función sirve para revisar que el método converga.
{
    int range = 10; //El rango de valores de la matriz y vector inicial aleatorio es 10.
    int itera = 10000; //2. a. Ocupamos un número fijo 1000 de iteraciones para toda aproximación del eigenvalor principal.
    double** A = matrix_generator(n, range);
    double* b0 = vector_generator(n, range);
    double* bk = power_iteration(A, b0, n, thre);
    double* result = (double*) calloc(100, sizeof(double));
    for(int i = 0; i < itera; i++)
    {
        bk = power_iteration(A, bk, n, thre);
        if(i > itera - 102)
        {
            result[i - (itera - 101)] = dominant_eig(A, bk, n, thre);
        }
    }
    free_matrix(A, n);
    free(b0);
    free(bk);
    return result;
}

double time_power(int n, int p, double T1) //2. f. Imprime el tiempo de cómputo, speedup y eficiencia (de p hilos) de 10000 iteraciones del método de la potencia. Adaptado del código en mat_vec.cpp de las ayudantías.
{
    auto inicio = std::chrono::high_resolution_clock::now();
    double* aprox_eig = thousand_eigs(n, p);
    auto fin = std::chrono::high_resolution_clock::now();
    auto tiempo = std::chrono::duration_cast<std::chrono::nanoseconds>(fin - inicio);
    free(aprox_eig);
    double Tp = tiempo.count() * 1e-9;
    printf("%.3f segundos \n", Tp);
    if (T1 == 1.0)
    {
        return Tp;
    }
    else
    {
        printf("Tenemos una speedup de valor %f \n", T1 / Tp);
        printf("y una eficiencia de valor %f \n", T1 / (p * Tp));
        return 0.0;
    }
}

int main()
{
    printf("Porfavor paciencia, una buena aproximacion tarda alrededor de 10 segundos \n");
    int thre = 2;
    int n = 2; //2. d. Revisamos que converga para una matriz pequeña.
    printf("Para n = %i tenemos la siguiente convergencia del metodo de la potencia \n", n);
    double* eigs = thousand_eigs(n, thre);
    print_arr(eigs, 100);
    n = 10; //Probamos la convergencia para matrizes de tamaños 10x10, 100x100 y 1000x1000.
    printf("Para n = %i tenemos la siguiente convergencia del metodo de la potencia \n", n);
    eigs = thousand_eigs(n, thre);
    print_arr(eigs, 100);
    thre = 6;
    n = 100;
    printf("Para n = %i tenemos la siguiente convergencia del metodo de la potencia \n", n);
    eigs = thousand_eigs(n, thre);
    print_arr(eigs, 100);
    thre = 30;
    n = 1000;
    printf("Para n = %i tenemos la siguiente convergencia del metodo de la potencia \n", n);
    eigs = thousand_eigs(n, thre);
    print_arr(eigs, 100);
    free(eigs);

    printf("Ciertas aproximaciones convergen de manera menos exacta porque puede suceder que la matriz generada aleatoriamente es muy poco sparse \n");
    printf("Porfavor paciencia de nuevo, con tan solo 1 hilo las 10000 iteraciones se hacen en alrededor de 20 o 30 segundos \n");

    thre = 1; //2. f. Calculamos el tiempo de cómputo de 10000 iteraciones del método de la potencia para 1, 6, 12, 24, 30 y 36 hilos.
    printf("Para %i hilos tenemos el siguiente tiempo de computo \n", thre);
    double T1 = time_power(n, thre, 1.0);
    thre = 6;
    printf("Para %i hilos tenemos el siguiente tiempo de computo \n", thre);
    double k = time_power(n, thre, T1);
    thre = 12;
    printf("Para %i hilos tenemos el siguiente tiempo de computo \n", thre);
    k = time_power(n, thre, T1);
    thre = 24;
    printf("Para %i hilos tenemos el siguiente tiempo de computo \n", thre);
    k = time_power(n, thre, T1);
    thre = 30;
    printf("Para %i hilos tenemos el siguiente tiempo de computo \n", thre);
    k = time_power(n, thre, T1);
    thre = 36;
    printf("Para %i hilos tenemos el siguiente tiempo de computo \n", thre);
    k = time_power(n, thre, T1);

    printf("Aqui podemos ver una aproximacion a la cantidad maxima de procesadores que podemos solicitar para el paralelismo");

    return 0;
}

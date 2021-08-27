#include <iostream>
#include <fstream>
#include <string>
#include "functions.cpp"
using namespace std;

int main() {
    // Primero, se generarán tres problemas,
    // de dimensiones 10, 100 y 1000.
    // Se testeará la convergencia de mu, corriendo
    // con 2 threads y 2000 iteraciones.

    for (int i = 1; i < 4; i++) {
        int n = pow(10, i);
        int t = 2;  // Número de threads.
        int iter = 2000;  // Número de iteraciones.

        float** A = matrix(n, 10);
        float* b = vector(n, 10);
        float* mu = (float*) calloc(iter, sizeof(float));

        run_algorithm(A, b, mu, n, iter, 2);

        // Exportamos un archivo con los distintos mu.
        string name = "mu" + to_string(n) + ".txt";
        ofstream MyFile(name);
        for (int k = 0; k < iter; k++) {
            MyFile << mu[k] << "\n";
        }
        MyFile.close();

        free_matrix(A, n);
        free(b);
        free(mu);
    }

    // Ahora, vemos el tiempo de cómputo con 1000
    // iteraciones, 1000 dimensiones y variando el número
    // de threads entre 0 (lineal) y 24.
    int n = 1000;
    int iter = 1000;  // Número de iteraciones.

    float** A = matrix(n, 10);
    float* b = vector(n, 10);
    float* mu = (float*) calloc(iter, sizeof(float));
    float* times = (float*) calloc(25, sizeof(float));

    for (int t = 0; t < 25; t++) {
        float* c = copy_vector(b, n);
        times[t] = return_time(A, c, mu, n, iter, t);
        free(c);    
    }

    // Exportamos un archivo con los distintos mu.
    ofstream MyFile("times.txt");
    for (int k = 0; k < 25; k++) {
        MyFile << times[k] << "\n";
    }
    MyFile.close();

    free_matrix(A, n);
    free(b);
    free(mu);
    free(times);

    return 0;
}

// Compilar y correr el archivo con el siguiente código:
// g++ -o statistics statistics.cpp -fopenmp ; ./statistics ; python3 plot_statistics.py

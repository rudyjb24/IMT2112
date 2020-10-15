#include <iostream>
#include <fstream>
#include "functions.cpp"
using namespace std;

int main() {

    int n = 1000;
    int t = 24;  // Número de threads.
    int iter = 1000;  // Número de iteraciones.

    float** A = matrix(n, 10);
    float* b = vector(n, 10);
    float* mu = (float*) calloc(iter, sizeof(float));

    run_algorithm(A, b, mu, n, iter, t);

    printf("Valor final mu: %f\n", mu[iter - 1]);

    // Exportamos un archivo con los distintos mu.
    ofstream MyFile("mu.txt");
    for (int k = 0; k < iter; k++) {
        MyFile << mu[k] << "\n";
    }
    MyFile.close();

    free_matrix(A, n);
    free(b);
    free(mu);
    return 0;
}

// Compilar y correr el archivo con el siguiente código:
// g++ -o main main.cpp -fopenmp ; ./main ; python3 plot_main.py

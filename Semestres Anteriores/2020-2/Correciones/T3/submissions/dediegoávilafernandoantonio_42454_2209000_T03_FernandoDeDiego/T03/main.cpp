#include <mpi.h>
#include <fstream>
#include "functions.cpp"
using namespace std;


int main() {
    MPI_Init(NULL, NULL);
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    char name[MPI_MAX_PROCESSOR_NAME];
    int len;
    MPI_Get_processor_name(name, &len);

    int local_rows, first_index, last_index, Nx, Ny, err;

    // Contamos los nodos borde.
    Nx = 1000;
    Ny = Nx;

    // Particionamos el dominio por filas.
    local_rows = Ny / world_size;
    first_index = world_rank * local_rows;
    
    if (world_rank == world_size - 1) {
        local_rows += Ny % world_size;
    }

    last_index = first_index + local_rows;

    printf("Rank: %i, World Size: %i\n Local rows: %i, First index %i, Last index: %i\n Name: ", 
        world_rank, world_size, local_rows, first_index, last_index);
    for (int i = 0; i < len; i++) {
        cout << name[i];
    }
    cout << "\n";

    double*** stencils = stencils_generator(first_index, last_index, Nx, Ny);

    // Acá guardaremos los resultados.
    double** x = array_generator(local_rows, Nx);
    double** r = array_generator(local_rows, Nx);
    double** z = array_generator(local_rows, Nx);
    double** q = array_generator(local_rows, Nx);
    double** p = array_generator(local_rows, Nx);
    
    double** result = array_generator(local_rows, Nx);

    double** b = puntual_function(local_rows, Nx, world_rank, world_size);
    // double** b = random_function(local_rows, Nx, world_rank, world_size);

    double* lower_neighbour = (double*) calloc(Nx, sizeof(double));
    double* upper_neighbour = (double*) calloc(Nx, sizeof(double));

    int iterations = 1000;

    double* remainder = (double*) calloc(iterations, sizeof(double));

    conjugate_gradient(stencils, x, r, z, result, q, p, b, lower_neighbour,
        upper_neighbour, remainder, world_rank, world_size, local_rows, Nx, iterations, err, Nx);

    vec_communication(x, world_rank, world_size, Nx, err, local_rows, lower_neighbour, upper_neighbour);

    if (world_rank == 0) {
        // Exportamos un archivo con los distintos residuos.
        ofstream MyFile("residuos.txt");
        for (int k = 0; k < iterations; k++) {
            MyFile << remainder[k] << "\n";
        }
        MyFile.close();

        // Printiamos el residuo final.
        cout << "Residuo final (iteración " << iterations << "): " << remainder[iterations - 1] << "\n";
        
        // Opcional: Printiemos el resultado. Debiese ser cercano a b.
        // print_matrix(b, local_rows, Nx);
        // mat_vec(stencils, x, lower_neighbour, upper_neighbour, result,
            // world_rank, world_size, local_rows, Nx);
        // print_matrix(result, local_rows, Nx);
    }

    free_stencils(stencils, local_rows);

    free_matrix(x, local_rows);
    free_matrix(r, local_rows);
    free_matrix(z, local_rows);
    free_matrix(q, local_rows);
    free_matrix(p, local_rows);

    free_matrix(result, local_rows);
    free_matrix(b, local_rows);
    
    free(lower_neighbour);
    free(upper_neighbour);
    free(remainder);

    MPI_Finalize();
    return 0;
}

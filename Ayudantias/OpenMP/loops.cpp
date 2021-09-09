#include <stdio.h>
#include <stdlib.h>
# include <omp.h>
 
int main() {
  
  int x = 0;

  #pragma omp parallel for num_threads(4)
  for (int i=0;i<10;i++) {
    int id = omp_get_thread_num();
    x += 1;
    printf("Proceso %d sumo 1\n", id);  
  }

  printf("\nx = %i\n", x);

  return 0;
}

































//  int n = 6;
//  int* array = (int *) calloc(n, sizeof(int));
//
//  printf("Tamaño %li \n", sizeof(float));
//  
//    #pragma omp parallel for num_threads(n)
//    for (int i=0;i<10;i++) {
//      int id = omp_get_thread_num();
//      array[id] += 1;
//      printf("Proceso %d sumo 1\n", id);  
//    }
//
//  for (int i=0;i<n;i++) {
//      x += array[i]; 
//    }
//
//  printf("\nx = %i\n", x);
//  }
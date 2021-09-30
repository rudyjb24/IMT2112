#include <stdio.h>
#include <stdlib.h>
# include <omp.h>
 
int main() {
  
  int x1 = 0;

  #pragma omp parallel for num_threads(24)
  for (int i=0;i<10000;i++) {
    int id = omp_get_thread_num();
    x1 += 1;
    printf("Proceso %d sumo 1\n", id);  
  }

  printf("\nx1 = %i\n", x1);




  int x2 = 0;
  int n = 10;
  int* array = (int*) calloc(n, sizeof(int));
  
  #pragma omp parallel for num_threads(n)
  for (int i=0;i<100;i++) {
    int id = omp_get_thread_num();
    array[id] += 1;
    printf("Proceso %d sumo 1\n", id);  
  }

  for (int i=0;i<n;i++) {
      x2 += array[i]; 
  }

  printf("\nx2 = %i\n", x2);




  return 0;
}
































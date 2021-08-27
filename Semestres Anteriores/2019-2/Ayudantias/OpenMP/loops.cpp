#include <stdio.h>
# include <omp.h>
 
int main() {
  
  int x = 0;
  {
    #pragma omp parallel //for
    for (int i=0;i<10;i++)
    {
      x += 1;
      //int id = omp_get_thread_num();
      //printf("Proceso %d sumo 1\n", id);  
    }

  printf("\nx = %i\n", x);
  }
  return 0;
}
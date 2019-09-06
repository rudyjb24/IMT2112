#include <stdio.h>
# include <omp.h>
 
int main() {
  #pragma omp parallel //num_threads(3)
  {
    printf("Hello World\n");
    int id = omp_get_thread_num();
    int total = omp_get_num_threads();
    //printf("\nGreetings from process %d out of %d \n", id, total);   
  }
  return 0;
}
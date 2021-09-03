#include <stdio.h>
#include <stdlib.h> 
#include <ctime>

int main() {
  
    int n = 10;
    int array1[n];
    
    int* array2 = (int*) calloc(10, sizeof(int)); // inicializa valores a 0

    int* array3 = (int*) malloc(10*sizeof(int)); // no inicializa valores

    for (int i=0;i<10;++i) {
        //printf("%i ", array3[i]);
    }
    

    srand((int) time(0));
    for (int i=0;i<10;++i) {
        //array2[i] = rand();
        //array2[i] = rand()%10;
        array2[i] = rand()%201 - 100;
        printf("%i ", array2[i]);
    }

    printf("\n");
    

    free(array2);
    free(array3);

    return 0;
}
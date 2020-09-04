#include <stdio.h>
#include <stdlib.h> 
#include <ctime>

int main() {
  
    int n = 10;
    int array1[n];
    
    int* array2 = (int*)calloc(10, sizeof(int)); //initializes memory block to 0

    for (int i=0;i<10;++i)
    {
        // printf("%i ", array1[i]);
    }
    
    printf("\n\n");
    srand((int)time(0));
    for (int i=0;i<10;++i)
    {
        //array2[i] = rand();
        //array2[i] = rand()%10;
        array2[i] = rand()%201 - 100;
        // printf("%i ", array2[i]);
    }

    printf("\n");
    

    free(array2);
    return 0;
}
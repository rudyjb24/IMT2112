#include <stdio.h>

int main() {
  
    int a = 1;
    int b; 
    b = 3;
    float c = 8;
    bool terminado = false;

    if (a+b > 2  && a%2 == 1) {
        printf("a+b = %i\n", a+b);
    }
    else if (a+b == 2 || c == 8) {
        printf("c = %f, %f\n", c, c);
    }
    else {

    }

    for (int i=0; i<10; i++) {
        a += 1;
    }
    printf("a = %i\n", a);


    while (!terminado) {
        b += 1;
        if (b % 4 == 0) {
            terminado = true;
        }
    }
    
    return 0;
}
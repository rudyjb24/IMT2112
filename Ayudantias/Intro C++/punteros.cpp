#include <stdio.h>

int main() {
  int a;
  int* b;

  a = 1;
  b = &a;
  
  a += 1;
  *b += 3; 

  printf("a = %i \n", a);
  printf("La direccion donde esta guardado a es: %p \n", &a);
  printf("b = %p \n", b);
  printf("El numero guardado en la direccion apuntada por b es: %i \n", *b);
   
  return 0;
}
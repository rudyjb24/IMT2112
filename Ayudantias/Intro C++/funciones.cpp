#include <stdio.h>

int suma(int x1, int x2) {
  x1 += 1;
  x2 += 2;
  return x1 + x2;
}

void actualizar(int* x) {
  *x = 1;
}

int main() {
  
  int a = 5;
  int b = 10;
  int c;

  printf("a = %i\n", a);
  printf("b = %i\n", b);
  printf("c = %i\n", c);

  return 0;
}













//c = suma(a,b);
//actualizar(&a);

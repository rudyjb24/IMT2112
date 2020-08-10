#include <stdio.h>

int suma(int x1, int x2)
{
  x1 += 1;
  x2 += 2;
  return x1 + x2;
}

void actualizar(int *x)
{
  *x = 1;
}

int main() {
  
  int a = 1;
  int b = 2;
  int c;

  c = suma(a,b);

  printf("c = %i \n", c);

  //actualizar(&a);

  //printf("a = %i \n", a);

  return 0;
}
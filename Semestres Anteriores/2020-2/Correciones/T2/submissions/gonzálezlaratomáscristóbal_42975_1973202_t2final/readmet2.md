La tarea está hecha completa.

a) Se usa metodo maximo de iteraciones (1000) o error 10^-4 entre u_k y u_k+1

b) Se libera toda la memoria pedida:

    Primero, en el main, se pide espacio para la matriz A y un vector b inicial.
    Se liberan al final del main.
    Luego, dentro de la función metodo_potencia, se calcula b_1 sin liberar
    el vector inicial, para que se pueda utilizar el mismo en las otras
    llamadas a la función (la llamamos para distinto numero de threads).
    Para calcular cada u_k se llama a mat_vec_par, que pide memoria, pero
    se libera dentro de la misma función u_k.
    Para calcular cada b_k, se usa un matvec donde se pide memoria, pero
    notemos que se libera en la iteración siguiente, ya que la función b_k
    recibe b_k-1 como parámetro 'vector' y en la misma función se hace un
    free(vector).

  Sin embargo, valgrind muestra memory leaks. Esto se debe a compilar con
  el flag -fopenmp. Link:
  https://medium.com/@auraham/pseudo-memory-leaks-when-using-openmp-11a383cc4cf9

c) Se usa el mat_vec_par de la ayudantía, y se paraleliza el producto punto
de manera thread safe, como aprendimos en ayudantia. Además, la norma no se
hace aparte, solo se calcula haciendo ||b|| = (prod_punto(b,b))^(.5).

d) Siempre se calcula la aproximación al valor propio mas grande. Se imprime
en consola cada vez. Se revisa que converge porque llevamos un registro de como
van cambiando los u_k en el tiempo y controlamos esa convergencia como se dijo
en la parte a) (revisando pequeña variación en mu después de una iteración).

e) El rendimiento del algoritmo es bueno hasta dimension 2000 más o menos con
1000 iteraciones. Luego de eso se vuelve un poco inmanejable. Con dimension
10.000 hay que ponerle que pare en 50 iteraciones, si no se demora mucho.
Se puede manejar este parametro al inicio del main.

f) Para revisar el tiempo de computo ejecutaremos el algoritmos con i hilos,
i entre 1 y 8, y ademas luego con 50 hilos. Los outputs son:

El algoritmo se ejecuto en dimension 1500
La cantidad de threads utilizados fue 2
Tiempo empleado: 6.066 segundos
Cantidad de iteraciones hechas: 441
Error de convergencia alcanzado: 0.000061
El valor absoluto del valor propio mas grande es aproximadamente: 446.774628



El algoritmo se ejecuto en dimension 1500
La cantidad de threads utilizados fue 3
Tiempo empleado: 5.160 segundos
Cantidad de iteraciones hechas: 437
Error de convergencia alcanzado: 0.000092
El valor absoluto del valor propio mas grande es aproximadamente: 446.772766



El algoritmo se ejecuto en dimension 1500
La cantidad de threads utilizados fue 4
Tiempo empleado: 5.612 segundos
Cantidad de iteraciones hechas: 446
Error de convergencia alcanzado: 0.000092
El valor absoluto del valor propio mas grande es aproximadamente: 446.776642



El algoritmo se ejecuto en dimension 1500
La cantidad de threads utilizados fue 5
Tiempo empleado: 5.483 segundos
Cantidad de iteraciones hechas: 461
Error de convergencia alcanzado: 0.000061
El valor absoluto del valor propio mas grande es aproximadamente: 446.781525



El algoritmo se ejecuto en dimension 1500
La cantidad de threads utilizados fue 6
Tiempo empleado: 4.844 segundos
Cantidad de iteraciones hechas: 455
Error de convergencia alcanzado: 0.000061
El valor absoluto del valor propio mas grande es aproximadamente: 446.779785



El algoritmo se ejecuto en dimension 1500
La cantidad de threads utilizados fue 7
Tiempo empleado: 6.229 segundos
Cantidad de iteraciones hechas: 471
Error de convergencia alcanzado: 0.000092
El valor absoluto del valor propio mas grande es aproximadamente: 446.784058



El algoritmo se ejecuto en dimension 1500
La cantidad de threads utilizados fue 50
Tiempo empleado: 7.171 segundos
Cantidad de iteraciones hechas: 479
Error de convergencia alcanzado: 0.000061
El valor absoluto del valor propio mas grande es aproximadamente: 446.785553


Notamos que se mejora hasta los 6 threads, luego con 7 empeora. Con 50 ya hay
muchos hilos que no hacen nada y se pierde mucho tiempo. Pero efectivamente al
principio la mejora es notoria.

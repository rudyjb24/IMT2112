# Readme tarea 2
## Vicente Hojas
* en el archivo tarea_2.cpp se encuentra el código de la tarea, 
con el ejecutable tarea_2.exe correspondiente.

* se realizan mediciones de tiempo para 2, 4, 8 y 12 threads y para matrices de 
n x n, con n en {10, 100, 1000, 10000}. Se miden los tiempos de ejecución 
para el método de la potencia secuencial y paralelo. Para 2 threads se ve que 
el tiempo disminuye a un poco más de la mitad y que para 4 threads o más el 
tiempo se reduce en la mayoría de los casos a un poco menos de la mitad. 
Para matrices pequeñas (n <= 1000) no hay mejora después de 4 threads. Para 
las matrices de 10.000 x 10.000 se consiguen mejoras más sustanciales 
(aun asi nunca un speed up de más de 1.5) sobre 4 threads.

* para matrices pequeñas (n = 10, 100), se demora más el algoritmo paralelo 
(presuntamente debido al header involucrado).

* Para asegurar la convergencia al crear la matriz la primera 
componente de la diagonal es 100 veces mayor al rango, de esta forma es casi 
seguro que la bola con centro A_00 que se ocupa en el teorema de Gershgorin no 
tenga intersección con el resto de las bolas, y por lo tanto es casi seguro 
que va a haber un solo valor propio alrededor de A_00 (valor muy grande), que 
entonces va a ser hasta 100 veces mayor que el segundo valor propio más grande y 
por lo tanto se asegura la convergencia rápida del algoritmo (recordar que 
depende de mu2/mu1).

* Por lo anterior, se observa en general un error relativo menor a 10^{-5} 
entre el algoritmo secuencial y paralelo, aun cuando el vector inicial en ambos 
casos es distinto (pues es aleatorio).

* No obstante lo anterior, para matrices muy grandes la convergencia asegurada 
por el teorema de Gershgorin puede fallar porque la bola es de radio igual 
a la suma de los valores absolutos de las componentes de las filas (spdg) 
y para una dimensión grande, aunque los valores individuales de cada componente
sean mucho menores que A_00, la suma puede ser de todas formas grande.

* nota: para las matrices de 10.000 x 10.000 el algoritmo se demora alrededor de 
5 minutos (con las 500 iteraciones que vienen en el código) si esto se quiere 
cambiar se puede hacer en main().
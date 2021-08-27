Tarea 2                                                                                              Williams Medina

Implementación:

La implementación del algoritmo fue realizada de forma análoga a la mostrada en el enunciado de la tarea. Para esta
se utilizaron las funciones vistas en la ayudantía, modificando la función de crear una matriz aleatoria de n x n 
para que fuera simétrica y de esta forma diagonalizable.

Para analizar la convergencia de los valores propios, guarde el valor obtenido en cada iteración y por inspección se 
puede ver desde que iteración este valor comienza a ser el mismo. Como es de esperar, la iteración en la que este 
comienza a converger es bastante aleatoria y depende demasiado de la dimensión de la matriz. Aunque para una cantidad 
muy grande de iteraciones el algoritmo siempre converge a un mu_k (encontrar cual es esa cant de iteraciones es el 
misterio).

Por alguna extraña razón, en algunas ejecuciones el código no me printeo nada en consola, pero si lo ejecutaba de nuevo
este si imprimía, intenté buscar el error, pero no encontré nada del estilo.

Testeos:

El código funciona en extremo rápido para dimensiones bajas y para dimensiones del orden 10.000 hay que variar demasiado
la cantidad de iteraciones para ver la convergencia del mu_k (además de que habría que revisar el valor obtenido en cada
iteración).

Para la cantidad de hilos probé con 4, 6 , 8 , 12, 16 y 24 (considerando que mi computador solo tiene 4 núcleos).


**Para dim = 100 y n_iteraciones = 100:

Threads = 4 Tiempo_ejecución = 0.039 seconds.
Threads = 6 Tiempo_ejecución = 0.057 seconds.
Threads = 8 Tiempo_ejecución = 0.070 seconds.
Threads = 12 Tiempo_ejecución = 0.102 seconds.
Threads = 16 Tiempo_ejecución = 0.128 seconds.
Threads = 24 Tiempo_ejecución = 0.187 seconds.

**Para dim = 100 y n_iteraciones = 1000:

Threads = 4 Tiempo_ejecución = 0.365 seconds.
Threads = 6 Tiempo_ejecución = 0.518 seconds.
Threads = 8 Tiempo_ejecución = 0.672 seconds.
Threads = 12 Tiempo_ejecución = 0.954 seconds.
Threads = 16 Tiempo_ejecución = 1.240 seconds.
Threads = 24 Tiempo_ejecución = 1.813 seconds.

**Para dim = 1000 y n_iteraciones = 1000:

Threads = 4 Tiempo_ejecución = 2.198 seconds.
Threads = 6 Tiempo_ejecución = 2.604 seconds.
Threads = 8 Tiempo_ejecución = 2.290 seconds.
Threads = 12 Tiempo_ejecución = 2.422 seconds.
Threads = 16 Tiempo_ejecución = 2.655 seconds.
Threads = 24 Tiempo_ejecución = 2.793 seconds.

**Para dim = 1000 y n_iteraciones = 10000

Threads = 4 Tiempo_ejecución = 22.434 seconds.
Threads = 6 Tiempo_ejecución = 28.091 seconds.
Threads = 8 Tiempo_ejecución = 30.529 seconds.
Threads = 12 Tiempo_ejecución = 29.262 seconds.
Threads = 16 Tiempo_ejecución = 32.438 seconds
Threads = 24 Tiempo_ejecución = 37.793 seconds.


**Conclusión

El tiempo de ejecución tiende a tener su funcionamiento óptimo en una cantidad de hilos que sea múltiplo de los cores del computador,
siendo para dimensiones más altas donde más se nota la diferencia de tiempo (en dim = 1.000 y n_iteraciones = 10.000) es donde más
notamos la diferencia, esto es lógico ya que la cantidad de iteraciones es mucho mayor tendiendo a ser con 4 threads el tiempo
de ejecución óptimo independiente de la dimensión y cantidad de iteraciones. Hecho completamente justificado considerando que mi computa-
dor tiene 4 núcleos, hecho muy importante que debe ser considerado a la hora de programar algoritmos paralelos, ya que si tomamos una cant
de threads mayor, la eficiencia disminuirá alejandonos de nuestro objetivo.
De todas formas, es importante notar que para dimensiones bajas, el tiempo de ejecución es bastante aleatorio para las milesimas de segundo
ya que puede que sean matrices más complejas que hagan que el código se demore 200 milesimas más que claramente son marginales a la hora de
comparar.

Con respecto a la convergencia, esta un poco más complicada de observar al tener una dimensión mayor, pero para dim 100 la convergencia se suele
obtener alrededor de las 50 iteraciones, para la dim 1000 esto varia bastante pero alrededor de las 300 se tiende a estabilizar. Esto de todas
formas es muy aleatorio ya que la matriz A y el vector b son aleatorios y varian entre cada ejecución.


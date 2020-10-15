Hola

Vengo a hacer un par de comentarios y no sabía como hacerlo asíque escribo este Readme.

De partida, los tiempos que me entregan no son los mejores, pues me dió lo "contrario" a lo esperado en términos de tiempo para
5000 datos, pero al aumentar el número se cambian los papeles solamente entre numpy y multiprocess que es lo esperado, pero creo
que esto se debe principalmente a la forma de calcular distancias que podría ocurrir que numpy tiene más precisión que la forma
en la que definí la norma y quizá por eso se demora mas el método "con loops".

Además sólo lo probé con 2 dimensiones (si varíe en cantidad de datos y centroides), me di cuenta de algo curioso que existen
distintos patrones estables de "centros" para datos homogeneos que se repetían, en un principio pensé que esto siempre convergía
a un único tipo de patrón pero no fue así. En realidad esto ultimo no tiene mucho que ver con lo que se pide pero me pareció curioso
de comentar.

Perdón por el formato de estos comentarios. En general traté de comentar todo el código y escribir nombres de variables intuitivas
para que fuese de fácil comprensión.

import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import random
import time


class KMeans:
    """
    Clase que ejecuta el algoritmo k-means implementado de distintas
    formas para llevar a cabo una comparación de cómputo.
    """
    def __init__(self, workers, conglomeraciones, data, max_iter=float("inf")):
        self.workers = workers  # Pool
        self.conglomeraciones = conglomeraciones  # k
        self.data = data  # X
        self.max_iter = max_iter
        # Ejecutamos todos los algoritmos y entregamos resultados
        self.ejecutar_algoritmo()

    def ejecutar_algoritmo(self):
        # Guardamos los resultados obtenidos en cada algoritmo (diccionarios)
        self.resultado_python_loops = self.python_loops()
        self.resultado_vectorized = self.vectorized()
        self.resultado_multiprocessing = self.multiprocessing()
        # Imprimimos los resultados
        self.imprimir_tiempos_ejecucion()

    def python_loops(self):
        """
        Algoritmo k-means, de manera que todas la iteraciones
        de k-means y cálculos están implementadas en bucles (un solo Thread).
        Correspondería a la versión más robusta en cuanto a código.
        """
        # Guardamos el tiempo inicial de ejecución del algoritmo
        tiempo_inicial = time.time()
        # Definimos las dimensiones en base a la data entregada
        n = len(self.data)
        m = len(self.data[0])
        # Definimos las listas que vamos a ir actualizando
        # y entregando como resultados
        centros = []
        etiquetas = [0 for i in range(n)]
        # En primer lugar, debemos seleccionar de manera
        # aleatoria los k centros iniciales
        indices = random.sample(range(n), self.conglomeraciones)
        for index in indices:
            centros.append(list(self.data[index]))
        # Ahora, entramos en el siguiente loop del algoritmo kmeans
        iteraciones = 0
        while self.max_iter != iteraciones:
            # (1) Tenemos que asignar una etiqueta a cada registro según distancia
            # euclidiana más cercana a algún centro
            for index, registro in enumerate(self.data):
                # Iteramos el cálculo de distancia en cada centro y vamos comparando
                distancia = float("inf")
                for etiqueta in range(self.conglomeraciones):
                    # Calculamos la distancia euclidiana
                    suma = 0
                    for coordenada in range(m):
                        suma += (centros[etiqueta][coordenada] - registro[coordenada]) ** 2
                    nueva_distancia = suma ** (1/2)
                    # Nos quedaremos siempre con la menor distancia posible
                    if nueva_distancia < distancia:
                        etiquetas[index] = etiqueta
                        distancia = nueva_distancia
            # (2) Ahora, hay que obtener los nuevos centros en base al centro
            # de masa de cada conglomeración.
            nuevos_centros = []
            for etiqueta in range(self.conglomeraciones):
                lista_sumas = [0 for k in range(m)]
                total = 0
                # Tenemos que sumar las coordenadas respectivas de todos los
                # registros que correspondan al cluster
                for index_registro, registro in enumerate(self.data):
                    # Aquí vemos si pertenece al cluster y entonces sumamos
                    if etiquetas[index_registro] == etiqueta:
                        for k in range(m):
                            lista_sumas[k] += registro[k]
                        total += 1
                # Ahora, obtenemos el nuevo centro como un promedio
                nuevo_centro = [0 for k in range(m)]
                for k in range(m):
                    nuevo_centro[k] = lista_sumas[k] / total
                nuevos_centros.append(nuevo_centro)
            # (3) Si los centros coinciden, entonces se termina el proceso.
            # En caso contrario se itera de nuevo considerando los nuevos centros.
            # (Puede terminar antes si se entregó un atributo "max_iter")
            centros_iguales = True
            for index in range(self.conglomeraciones):
                for k in range(m):
                    if centros[index][k] != nuevos_centros[index][k]:
                        centros_iguales = False
            if centros_iguales:
                break
            centros = nuevos_centros.copy()
            iteraciones += 1
        # Guardamos el tiempo final de ejecución y calculamos el tiempo total
        tiempo_final = time.time()
        tiempo_ejecucion = tiempo_final - tiempo_inicial
        # Ahora, retornamos los resultados en un diccionario
        resultado = {
            "centros": centros, "etiquetas": etiquetas,
            "ejecucion": tiempo_ejecucion, "iteraciones": iteraciones}
        return resultado

    def vectorized(self):
        """
        Algoritmo k-means implementado y vectorizado con NumPy.
        De esta forma, se usará más de un Thread.
        """
        # Guardamos el tiempo inicial de ejecución del algoritmo
        tiempo_inicial = time.time()
        # Definimos las dimensiones en base a la data entregada
        n, m = np.shape(self.data)
        # Definimos los centros inicial (random) y array de etiquetas
        indices = np.random.permutation(n - 1)[:self.conglomeraciones]
        centros = self.data[indices]
        etiquetas = np.zeros(n)
        # Ahora, entramos en el siguiente loop del algoritmo kmeans
        iteraciones = 0
        while self.max_iter != iteraciones:
            # (1) Tenemos que asignar una etiqueta a cada registro según distancia
            # euclidiana más cercana a algún centro
            for index, registro in enumerate(self.data):
                distancias = np.array(
                    [np.linalg.norm(centros[i] - registro) for i in range(self.conglomeraciones)])
                etiquetas[index] = np.argmin(distancias)
            # (2) Ahora, hay que obtener los nuevos centros en base al centro
            # de masa de cada conglomeración.
            nuevos_centros = np.empty((self.conglomeraciones, m))
            for etiqueta in range(self.conglomeraciones):
                nuevos_centros[etiqueta] = (
                    np.mean(self.data[etiquetas == etiqueta], axis=0))
            # (3) Si los centros coinciden, entonces se termina el proceso.
            # En caso contrario se itera de nuevo considerando los nuevos centros.
            # (Puede terminar antes si se entregó un atributo "max_iter")
            if np.all(centros == nuevos_centros):
                break
            centros = np.copy(nuevos_centros)
            iteraciones += 1
        # Guardamos el tiempo final de ejecución y calculamos el tiempo total
        tiempo_final = time.time()
        tiempo_ejecucion = tiempo_final - tiempo_inicial
        # Ahora, retornamos los resultados en un diccionario
        resultado = {
            "centros": centros, "etiquetas": etiquetas,
            "ejecucion": tiempo_ejecucion, "iteraciones": iteraciones}
        return resultado

    def multiprocessing(self):
        """
        Algoritmo k-means implementado con multiprocessing.
        En particular, usando la función Pool(workers).
        """
        # Guardamos el tiempo inicial de ejecución del algoritmo
        tiempo_inicial = time.time()
        # Definimos las dimensiones en base a la data entregada
        n, m = np.shape(self.data)
        # Definimos los centros inicial (random) y array de etiquetas
        indices = np.random.permutation(n - 1)[:self.conglomeraciones]
        self.centros = self.data[indices]  # Tiene que ser guardado en la clase
        etiquetas = np.zeros(n)
        # Ahora, entramos en el siguiente loop del algoritmo kmeans
        iteraciones = 0
        while self.max_iter != iteraciones:
            # Instanciamos la clase Pool
            pool = Pool(self.workers)
            # Luego, vamos a definir los chunks de los registros que vamos a
            # ir asignando a los workers.
            size_chunks = int(n / self.workers)
            chunks = []
            for i in range(self.workers - 1):
                chunks.append(self.data[i * size_chunks:(i + 1) * size_chunks])
            chunks.append(self.data[(i + 1) * size_chunks:])
            # (1) Tenemos que asignar una etiqueta a cada registro según distancia
            # euclidiana más cercana a algún centro
            chunks_etiquetas = list(
                pool.map(self.multiprocess_asignar_etiqueta, chunks))
            etiquetas = np.concatenate(
                list(pool.map(self.multiprocess_asignar_etiqueta, chunks)))
            # Vamos a mezclar ambos chunks en uno solo
            chunks_data = []
            for i in range(self.workers):
                chunks_data.append(
                    list(zip(chunks[i], chunks_etiquetas[i])))
            # (2) Ahora, hay que obtener los nuevos centros en base al centro
            # de masa de cada conglomeración.
            nuevos_centros = np.empty((self.conglomeraciones, m))
            for etiqueta in range(self.conglomeraciones):
                self.etiqueta_actual = etiqueta
                lista_tuplas = pool.map(self.multiprocess_obtener_suma, chunks_data)
                total = sum([tupla[0] for tupla in lista_tuplas])
                lista_sumas = np.zeros(m)
                for k in range(m):
                    lista_sumas[k] += sum([tupla[1][k] for tupla in lista_tuplas])
                nuevo_centro = np.array(
                    [suma / total for suma in lista_sumas])
                nuevos_centros[etiqueta] = nuevo_centro
            # (3) Si los centros coinciden, entonces se termina el proceso.
            # En caso contrario se itera de nuevo considerando los nuevos centros.
            # (Puede terminar antes si se entregó un atributo "max_iter")
            if np.all(self.centros == nuevos_centros):
                break
            self.centros = np.copy(nuevos_centros)
            iteraciones += 1
        # Guardamos el tiempo final de ejecución y calculamos el tiempo total
        tiempo_final = time.time()
        tiempo_ejecucion = tiempo_final - tiempo_inicial
        # Ahora, retornamos los resultados en un diccionario
        resultado = {
            "centros": self.centros, "etiquetas": etiquetas,
            "ejecucion": tiempo_ejecucion, "iteraciones": iteraciones}
        return resultado

    def multiprocess_asignar_etiqueta(self, registros):
        etiquetas = np.zeros(len(registros))
        for index, registro in enumerate(registros):
            distancias = np.array(
                [np.linalg.norm(self.centros[i] - registro) for i in range(self.conglomeraciones)])
            etiquetas[index] = np.argmin(distancias)
        return etiquetas

    def multiprocess_obtener_suma(self, data):
        total = 0
        lista_sumas = np.zeros(len(data[0][0]))
        for registro, etiqueta in data:
            if int(etiqueta) == self.etiqueta_actual:
                for k in range(len(registro)):
                    lista_sumas[k] += registro[k]
                total += 1
        return total, lista_sumas

    def imprimir_tiempos_ejecucion(self):
        """
        Imprime los tiempos de ejecución en consola.
        """
        tiempo_1 = self.resultado_python_loops["ejecucion"]
        tiempo_2 = self.resultado_vectorized["ejecucion"]
        tiempo_3 = self.resultado_multiprocessing["ejecucion"]
        print(
            "\nTiempo de ejecución algoritmo Python Loops: " +
            str(tiempo_1) + " [s] !!!")
        print(
            "\nTiempo de ejecución algoritmo Vectorizado: " +
            str(tiempo_2) + " [s] !!!")
        print(
            "\nTiempo de ejecución algoritmo Multiprocessing: " +
            str(tiempo_3) + " [s] !!!\n")

    def graficar_conglomeraciones(self):
        """
        Retorna los scatter plot de cada algoritmo k-means respectivo,
        junto con el tiempo de ejecución.
        """
        # Esta función no está implementada para m > 2 !!!! D:
        n, m = np.shape(self.data)
        # Algoritmo k-means implementado con Python Loops

        etiquetas = self.resultado_python_loops["etiquetas"]
        centros = self.resultado_python_loops["centros"]
        tiempo_ejecucion = self.resultado_python_loops["ejecucion"]
        iteraciones = self.resultado_python_loops["iteraciones"]

        plt.figure()

        plt.scatter(
            self.data[:, 0], self.data[:, 1], c=etiquetas,
            s=50, cmap='viridis')
        for k in range(self.conglomeraciones):
            plt.scatter(
                centros[k][0], centros[k][1], c='white', marker="o",
                s=180, edgecolors="black")
            plt.scatter(
                centros[k][0], centros[k][1], c='black',
                marker=f"${k + 1}$", s=60)
            plt.title(
                "Algoritmo k-means implementado con Python Loops" +
                "\nTiempo de ejecución: " + str(tiempo_ejecucion) + " [s]" +
                "\nNúmero de iteraciones: " + str(iteraciones))
            plt.xlabel(
                "Registros: " + str(n) +
                "   &   Dimensión: " + str(m))

        # Algoritmo k-means implementado con NumPy

        etiquetas = self.resultado_vectorized["etiquetas"]
        centros = self.resultado_vectorized["centros"]
        tiempo_ejecucion = self.resultado_vectorized["ejecucion"]
        iteraciones = self.resultado_vectorized["iteraciones"]

        plt.figure()

        plt.scatter(
            self.data[:, 0], self.data[:, 1], c=etiquetas,
            s=50, cmap='viridis')
        for k in range(self.conglomeraciones):
            plt.scatter(
                centros[k][0], centros[k][1], c='white', marker="o",
                s=180, edgecolors="black")
            plt.scatter(
                centros[k][0], centros[k][1], c='black',
                marker=f"${k + 1}$", s=60)
            plt.title(
                "Algoritmo k-means implementado con vectorización de NumPy" +
                "\nTiempo de ejecución: " + str(tiempo_ejecucion) + " [s]" +
                "\nNúmero de iteraciones: " + str(iteraciones))
            plt.xlabel(
                "Registros: " + str(n) +
                "   &   Dimensión: " + str(m))

        # Algoritmo k-means implementado con multiprocessing

        etiquetas = self.resultado_multiprocessing["etiquetas"]
        centros = self.resultado_multiprocessing["centros"]
        tiempo_ejecucion = self.resultado_multiprocessing["ejecucion"]
        iteraciones = self.resultado_multiprocessing["iteraciones"]

        plt.figure()

        plt.scatter(
            self.data[:, 0], self.data[:, 1], c=etiquetas,
            s=50, cmap='viridis')
        for k in range(self.conglomeraciones):
            plt.scatter(
                centros[k][0], centros[k][1], c='white', marker="o",
                s=180, edgecolors="black")
            plt.scatter(
                centros[k][0], centros[k][1], c='black',
                marker=f"${k + 1}$", s=60)
            plt.title(
                "Algoritmo k-means implementado con multiprocessing" +
                "\nTiempo de ejecución: " + str(tiempo_ejecucion) + " [s]" +
                "\nNúmero de iteraciones: " + str(iteraciones))
            plt.xlabel(
                "Registros: " + str(n) +
                "   &   Dimensión: " + str(m) +
                "   &   Workers: " + str(self.workers))

        # Mostramos todos los gráficos

        plt.show()


if __name__ == "__main__":
    # Tenemos que importar n >= 1000 registros de m >= 2 dimensiones.
    # A estos registros le haremos un clustering dado por k >= 3,
    # el número de conglomeraciones.
    m = 2
    n = 1000
    X = np.random.rand(n, m)
    # Ahora, ejecutamos el algoritmo k-means y entregamos todos los resultados.
    kmeans = KMeans(workers=4, conglomeraciones=12, data=X, max_iter=float("inf"))
    # Se muestra el gráfico
    kmeans.graficar_conglomeraciones()

    # #### OBSERVACIONES

    # Los primeros dos algoritmos funcionan bastante bien a
    # comparación del de multiprocessing, el cual no pude
    # paralelizar bien debido a que al solo podía repartir
    # los bucles y terminaba ejecutando varias veces
    # Pool en la parte de calcular los nuevos centros (la
    # cantidad de ejecuciones depende ampliamente del
    # número de clusters, lo que podría estar afectando
    # negativamente el tiempo de ejecución)

    # El método de clustering k-means implementado es bastante
    # sensible debido a la selección random de centros iniciales,
    # de manera que el número de iteraciones y resultado puede
    # cambiar considerablemente aun teniendo las mismas
    # variables iniciales al ejecutar nuevamente el algoritmo.
    # Por este motivo, se estableción un atributo opcional
    # que es el de las máximas iteraciones posibles, para así
    # comparar de manera más precisa los tiempos de ejecución.

    # Si mantenemos una dimensión pequeña y una cantidad
    # alta de registros, entonces el algoritmo python loops
    # es más rápido que el vectorizado. Sin embargo, si
    # vamos aumentando más la dimensión (por ejemplo,
    # m = 15) entonces el algoritmo vectorizado ya empieza
    # a ser más rápido, debido al uso eficiente de la memoria.

    # No logré reducir la dimensionalidad de los registros
    # para graficar los scatter plots, de manera que solo
    # funciona para m = 2

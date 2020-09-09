import sklearn.datasets as data
import math
import time
import random
import numpy as np
from multiprocessing import Pool
from functools import partial
import matplotlib.pyplot as plt


def calcular_distancia(a, b):
    dist = 0
    for i in range(len(a)):
        dist += (a[i] - b[i]) ** 2
    dist = math.sqrt(dist)
    return dist


def k_means_normal(k, muestras, centros):
    terminar = False
    indices_old = [0 for i in range(0, len(muestras))]
    itera = 0
    while not terminar:
        i = 0
        indices_new = [0 for i in range(0, len(muestras))]
        for muestra in muestras:
            distancias = []
            for centro in centros:
                d = calcular_distancia(muestra, centro)
                distancias.append(d)
            indices_new[i] = distancias.index(min(distancias))
            i += 1
        if indices_new == indices_old:
            terminar = True
        else:
            indices_old = indices_new
            centros_new = []
            for j in range(0, len(centros)):
                centro = []
                for dim in range(len(muestras[0])):
                    suma_dim = 0
                    elem_dim = 0
                    for _ in range(len(muestras)):
                        if indices_new[_] == j:
                            suma_dim += muestras[_][dim]
                            elem_dim += 1
                    if elem_dim == 0:
                        break
                    centro.append(suma_dim/elem_dim)
                centros_new.append(tuple(centro))
        itera += 1
    print(indices_new)
    print()
    print(indices_old)
    return centros, indices_new, itera


def k_means_vectorizado(k, muestras, centros):
    terminar = False
    indices_old = np.zeros(len(muestras))
    indices_new = np.zeros(len(muestras))
    itera = 0
    while not terminar:
        i = 0
        for muestra in muestras:
            distancias = np.zeros(k)
            for j in range(len(centros)):
                d = np.linalg.norm(centros[j]-muestra)
                distancias[j] = d
            indices_new[i] = np.argmin(distancias)
            i += 1
        if (indices_new == indices_old).all():
            terminar = True
        else:
            indices_old = indices_new
            for j in range(0, len(centros)):
                suma_actual = np.zeros(len(muestras[0]))
                total = 0
                for h in range(len(muestras)):
                    if indices_old[h] == j:
                        suma_actual += muestras[h]
                        total += 1
                if total == 0:
                    break
                centros[j] = suma_actual / total
        itera += 1
    return centros, indices_new, itera


def encontrar_indice(centros, punto):
    distancias = np.zeros(len(centros))
    for j in range(len(centros)):
        d = np.linalg.norm(centros[j]-punto)
        distancias[j] = d
    return np.argmin(distancias)


def separar_indices(muestras, indices, k):
    # esta funcion devuelve una lista de listas con las muestras que estan en un mismo indice.
    # asi, listas[0] son todas las muestra que pertenecen al conglomerado 0.
    listas = []
    for i in range(k):
        listas.append([])
    for j in range(len(indices)):
        listas[indices[j]].append(muestras[j])
    return listas


def encontrar_centro(conglomeracion):
    if conglomeracion:
        suma = sum(conglomeracion)
        centro = suma / len(conglomeracion)
        return centro
    else:
        return "este conglomerado está vacío"


def k_means_paralelizado(k, muestras, centros, p):
    terminar = False
    indices_old = np.zeros(len(muestras))
    pool = Pool(p)
    itera = 0
    while not terminar:
        func = partial(encontrar_indice, centros)
        indices_new = np.array(pool.map(func, muestras))
        if (indices_new == indices_old).all():
            terminar = True
        else:
            indices_old = indices_new
            listas = separar_indices(muestras, indices_new, len(centros))
            centros_ = pool.map(encontrar_centro, listas)
            centros = []
            for centro in centros_:
                if not isinstance(centro, str):
                    centros.append(centro)
        itera += 1
    return centros, indices_new, itera


def comparar(dim, n_muestras, k, trabajadores=2):
    # generamos los datos:
    x, y = data.make_blobs(n_samples=n_muestras, n_features=dim, centers=k)

    # creo los centros iniciales
    centros_inicio = random.choices(x, k=k)

    print(f"Vamos a comparar el algoritmo aplicado a {n_muestras} muestras, con {dim} parametros cada una:\n")

    # normal:
    t_1 = time.time()
    centros, indices, itera = k_means_normal(k, x, np.copy(centros_inicio))
    t_2 = time.time()
    print(f"El tiempo que toma k-means sin vectorizar ni paralelizar es {t_2-t_1} segundos.")
    print(f"Para esto realiza {itera} iteraciones")

    # vectorizando:
    # vale mencionar que vectorizar solo es mas rapido al tener muchas dimensiones.
    # con menos de 5 no vale la pena y hasta es mas lento.
    t_1 = time.time()
    centros, indices, itera = k_means_vectorizado(k, x, np.copy(centros_inicio))
    t_2 = time.time()
    print(f"El tiempo que toma k-means vectorizado pero sin paralelizar es {t_2-t_1} segundos.")
    print(f"Para esto realiza {itera} iteraciones")

    # paralelizando:
    # si bien logra su objetivo, este algoritmo NO es mas rapido que normal o vectorizado.
    # una de las razones por las que esto pasa es porque toma muchas mas iteraciones en
    # lograr que los indices se mantengan constantes. 
    t_1 = time.time()
    centros, indices, itera = k_means_paralelizado(k, x, np.copy(centros_inicio), trabajadores)
    t_2 = time.time()
    print(f"El tiempo que toma k-means paralelizado con {trabajadores} workers es {t_2-t_1} segundos.")
    print(f"Para esto realiza {itera} iteraciones\n")

    print("*" * 30 + "\n")

    return centros, indices, x


if __name__ == "__main__":
    centros, indices, x = comparar(2, 1000, 5)
    centros, indices, x = comparar(10, 1000, 5)
    centros, indices, x = comparar(10, 10000, 5, 4)
    centros, indices, x = comparar(10, 10000, 5, 8)

    # hacemos el scatter plot
    X = np.array(list(map(lambda p: p[0], x)))
    Y = np.array(list(map(lambda p: p[1], x)))
    plt.scatter(X, Y, c=indices)

    # ponemos los centros
    x_centros = np.array(list(map(lambda p: p[0], centros)))
    y_centros = np.array(list(map(lambda p: p[1], centros)))
    plt.scatter(x_centros, y_centros, s=200, c="magenta")
    plt.savefig("scatterplot.jpg")

    # a veces se ven puntos de colores distintos mezclados en el plot.
    # esto es porque solo estamos viendo dos dimensiones de 10.

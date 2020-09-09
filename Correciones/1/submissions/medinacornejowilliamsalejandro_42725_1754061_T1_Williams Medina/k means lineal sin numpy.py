from sklearn.datasets import make_blobs
from itertools import repeat
import time as tm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def generar_datos(n_filas, n_columnas):
    datos, categorias = make_blobs(n_samples=n_filas, n_features=n_columnas, random_state=0)
    return(datos)


def calcular_centroide(datos, dim):
    centros = [0] * dim
    if len(datos) !=0:
        for dato in datos:
            for i in range(dim):
                centros[i] += dato[i]/len(datos)
        return(centros)

    else:
        return (centros)


def calcular_distancia(punto, centro):
    valor = 0
    for punto, centro in zip(punto, centro):
        valor += (punto - centro)**2
    return (float(valor**(1/2)))


def asignar_label(punto, centros):
    distancia = 10000
    min = 0
    for i in range(len(centros)):
        d = calcular_distancia(punto, centros[i])
        if d < distancia:
            distancia = d
            min = i
    return([punto,min])

if __name__ == '__main__':
    def k_means(k, datos):
        dim = len(datos[0])
        tamaño = int(len(datos)/k)
        lista_grupos = []
        for i in range(k):
            lista_grupos.append(datos[i*tamaño:(i+1)*tamaño])
        centroides = []
        for grupo in lista_grupos:
            centroides.append(calcular_centroide(grupo, dim))
        while True:
            lista_labels = []
            for punto in datos:
                lista_labels.append(asignar_label(punto,centroides))

            lista_final = []
            for i in range(len(centroides)):
                lista_centro = []
                for punto in lista_labels:
                    if punto[1] ==  i:
                        lista_centro.append(punto[0])
                lista_final.append(lista_centro)
            nuevos_centroides = []
            for grupo in lista_final:
                nuevos_centroides.append(calcular_centroide(grupo, dim))

            if np.allclose(nuevos_centroides, centroides):
                break
            centroides = nuevos_centroides
        return (centroides, lista_labels)


    """Ploteamos para 2 dimensiones y un k=4
    datos =  generar_datos(1000, 2)
    k = 4
    centros, puntos = k_means(k, datos)
    centros = np.array(centros)
    print(centros)
    puntos = np.array(puntos)
    labels = np.array(puntos)[:, 1]
    colores=['red','green','blue','cyan']
    colores_2 = ["green",'red','cyan','blue']
    asignar=[]
    for row in labels:
        asignar.append(colores[row])
    plt.scatter(datos[:, 0], datos[:, 1], c = asignar, s=50);
    plt.scatter(centros[:,0], centros[:,1], marker='*', c=colores_2, s=100)
    plt.show()"""

    """Para 3 dimensiones y un k=5"""
    datos =  generar_datos(100000, 3)
    k = 12
    t0 = tm.time()
    centros, puntos = k_means(k, datos)
    t1 = tm.time()
    print('Tiempo de ejecución: ', t1-t0)
    """centros = np.array(centros)
    puntos = np.array(puntos)
    labels = np.array(puntos)[:, 1]
    colores=["yellow",'red','green','blue','cyan']
    colores_2 = ["green",'red','cyan','blue', "yellow"]
    asignar=[]
    for row in labels:
        asignar.append(colores[row])
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(datos[:, 0], datos[:, 1], datos[:,2], c = asignar, s=40);
    ax.scatter(centros[:,0], centros[:,1], centros[:,2], marker='*', c=colores_2, s=100)
    plt.show()"""

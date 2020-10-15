from sklearn.datasets import make_blobs
from itertools import repeat
import multiprocessing
import numpy as np
import time as tm
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

def generar_datos(n_filas, n_columnas):
    datos, categorias = make_blobs(n_samples=n_filas, n_features=n_columnas, random_state=0)
    return(np.array(datos))


def calcular_centroide(datos):
    dim = 3
    if len(datos) !=0:
        dim = len(datos[0])
        return (np.mean(datos, axis=0))
    else:
        listofzeros = [0] * dim
        return (np.array(listofzeros))


def calcular_distancia(punto, centro):
    valor = 0
    for punto, centro in zip(punto, centro):
        valor += (punto - centro)**2
    return (float(np.sqrt(valor)))


def asignar_label(punto, centros):
    distancia = 10000
    min = 0
    for i in range(len(centros)):
        d = calcular_distancia(punto, centros[i])
        if d < distancia:
            distancia = d
            min = i
    return([punto,min])


def agrupar(punto, centros, label):
    for i in range(len(centros)):
        if punto[1] == i:
                return(punto)


if __name__ == '__main__':
    def k_means(k, datos):
        n_workers = 8
        p =  multiprocessing.Pool(n_workers)
        tamaño = int(len(datos)/k)
        lista_grupos = []
        for i in range(k):
            lista_grupos.append(datos[i*tamaño:(i+1)*tamaño])
        centroides = list(p.map(calcular_centroide, lista_grupos))
        while True:
            punto_0 = 0
            lista_labels = list(p.starmap(asignar_label, zip(datos, repeat(centroides))))
            lista_final = []
            for i in range(len(centroides)):
                lista_centro = []
                for punto in lista_labels:
                    if punto[1] ==  i:
                        lista_centro.append(punto[0])
                lista_final.append(lista_centro)
            nuevos_centroides = list(p.map(calcular_centroide, lista_final))
            if all([np.allclose(x, y) for x, y in zip(nuevos_centroides, centroides)]):
                break
            centroides = nuevos_centroides
        return (centroides, lista_labels)

    #Ploteamos para 2 dimensiones y un k=4
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
    plt.show()

    """Para 3 dimensiones y un k=5
    datos =  generar_datos(1000, 3)
    k = 5
    t0 = tm.time()
    centros, puntos = k_means(k, datos)
    t1 = tm.time()
    print('Tiempo de ejecución: ', t1-t0)
    centros = np.array(centros)
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

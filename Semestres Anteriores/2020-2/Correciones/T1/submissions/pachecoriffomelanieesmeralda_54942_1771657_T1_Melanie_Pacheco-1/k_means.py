import numpy as np
import matplotlib.pyplot as plt
import time as tm
import multiprocessing
import math
from sklearn.decomposition import PCA
import pandas as pd


#Dimensión m y Registro n
m = 2
n = 1000

#Generar datos
datos = np.random.rand(n, m)

#Cantidad centros iniciales k
k = 7
print(f"Dimensión: {m} \nRegistro: {n} \nCantidad centros: {k}")

#Generador de centros
centros_incial = np.random.rand(k, m)
centros = centros_incial.copy()

# tolerancia entre distancia de centros
tolerancia = 0.00001

# Parte (a) PythonLoops

#calcula distancia entre dos puntos de misma dimension
def distancia(x, y):
    contador = 0
    for i in range(len(x)):
        contador += (x[i] - y[i])**2
    contador = (contador)**(1/2)
    return contador


def generador_cluster(datos, centros):
    clusters = list([] for i in range(len(centros)))
    for dato in datos:
        d_menor = -1
        for i in range(len(centros)):
            d = distancia(dato, centros[i])
            if d_menor == -1 or d < d_menor:
                d_menor = d
                centro = i
        clusters[centro].append(dato)
    return clusters

def suma(vectores):

    dimension = len(vectores[0])
    suma_total = np.zeros(dimension)

    for i in range(dimension):
        suma_filas = 0
        for j in range(len(vectores)):
            suma_filas += vectores[j][i]
        suma_total[i] += suma_filas

    return suma_total

def centros_masa(clusters):
    centros = []
    for cluster in clusters:
        centro = suma(cluster) / len(cluster)
        centros.append(centro)
    return centros

t_inicial = tm.time()
centro_encontrado = False

while not centro_encontrado:
    centro_encontrado = True
    clusters = generador_cluster(datos, centros)
    centros_nuevos = centros_masa(clusters)
    for i in range(len(centros_nuevos)):
        d = distancia(centros_nuevos[i], centros[i])
        if d >= tolerancia:
            centro_encontrado = False
    centros = centros_nuevos
t_final = tm.time()

print("Tiempo en realizar parte (a): ", t_final - t_inicial)

# Parte (b) uso de numpy

def generador_cluster_numpy(datos, centros):
    clusters = list([] for i in range(len(centros)))
    for dato in datos:
        d_menor = -1
        for i in range(len(centros)):
            d = np.linalg.norm(dato - centros[i])
            if d_menor == -1 or d < d_menor:
                d_menor = d
                centro = i
        clusters[centro].append(dato)
    return clusters

def centros_masa_numpy(clusters):
    centros = []
    for cluster in clusters:
        centro = np.sum(cluster, axis = 0) / len(cluster)
        centros.append(centro)
    return centros

centros = centros_incial.copy()
t_inicial = tm.time()
centro_encontrado = False

while not centro_encontrado:
    centro_encontrado = True
    clusters = generador_cluster_numpy(datos, centros)
    centros_nuevos = centros_masa_numpy(clusters)
    for i in range(len(centros_nuevos)):
        d = np.linalg.norm(centros_nuevos[i] - centros[i])
        if d >= tolerancia:
            centro_encontrado = False
    centros = centros_nuevos
t_final = tm.time()

print("Tiempo en realizar parte (b): ", t_final - t_inicial)



# Parte (c) uso de multiprocessing


def centros_masa_multiprocessing(clusters):
    centros = []
    for cluster in clusters:
        centro = np.sum(cluster, axis = 0) / len(cluster)
        centros.append(centro)
    return centros

def generador_cluster_multiprocessing(data):
    datos = data[0]
    centros = data[1]
    clusters = list([] for i in range(len(centros)))
    for dato in datos:
        d_menor = -1
        for i in range(len(centros)):
            d = np.linalg.norm(dato - centros[i])
            if d_menor == -1 or d < d_menor:
                d_menor = d
                centro = i
        clusters[centro].append(dato)
    return clusters

centros = centros_incial.copy()
n_trabajadores = 8

p = multiprocessing.Pool(n_trabajadores)

t_inicial = tm.time()
centro_encontrado = False

while not centro_encontrado:
    centro_encontrado = True

    # multiprocessing generador de cluster "gc"
    size_chuncks_gc = math.ceil(n / n_trabajadores)
    chuncks_gc = []

    for i in range(n_trabajadores-1):
        chuncks_gc.append([list(datos[i*size_chuncks_gc:(i+1)*size_chuncks_gc]), centros])

    chuncks_gc.append([datos[(i+1)*size_chuncks_gc:], centros])

    clusters_nuevos = list(p.map(generador_cluster_multiprocessing, chuncks_gc))
    
    clusters = list([] for i in range(k))

    for cluster_nuevo in clusters_nuevos:
        for i in range(k):
            clusters[i] += cluster_nuevo[i]

    # multiprocessig centros de masa "cm"
    size_chuncks_cm = math.ceil(k / n_trabajadores)
    chuncks_cm = []

    for i in range(n_trabajadores-1):
        chuncks_cm.append(clusters[i*size_chuncks_cm:(i+1)*size_chuncks_cm])

    chuncks_cm.append(clusters[(i+1)*size_chuncks_cm:])
    centros_nuevos_map = list(p.map(centros_masa_multiprocessing, chuncks_cm))
    centros_nuevos = []

    for centro in centros_nuevos_map:
        centros_nuevos += centro

    for i in range(len(centros_nuevos)):
        d = np.linalg.norm(centros_nuevos[i] - centros[i])
        if d >= tolerancia:
            centro_encontrado = False
    centros = centros_nuevos
t_final = tm.time()

print("Tiempo en realizar parte (c): ", t_final - t_inicial)


pca = PCA(n_components=2)

for cluster in clusters:
    x, y = zip(*cluster)
    plt.scatter(x, y)

x, y = zip(*centros)
plt.scatter(x, y, s=180, c="black", alpha=0.9, marker=r'$\clubsuit$',
            label="Luck")
plt.show()


## Se puede demorar menos el primero ya que la dimensionalidad es 2, si se aumenta la 
## dimensionalidad ve observa una diferencia considerable
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from numpy.random import rand

from multiprocessing import Process, Pool
import time as tm

def kmeans(data, seeds):
    # Función que implementa el algoritmo K-means
    # Retorna un vector
    
    # Variables:
    # data: matriz de nxd que contiene vectores a clusterizar; n es el número de vectores y d es su dimensión
    # n_clust: número de clusters deseados
    # seeds: centroides ("semillas") iniciales
    # max_iter: número máximo de iteraciones permitidas como criterio de convergencia

    n_clust = 4

    label = [] # Lista de etiqueta para clusters
    centroids = np.zeros((n_clust,2)) # Centroides de los clusters etiquetados por orden creciente
        
    # Iteraciones
    for i in range(100):
        # Asociacion a clústers
        for row_data in data:
            aux = [norm(row_data - row_seeds) for row_seeds in seeds]
            label.append(aux.index(min(aux)))
        # Recálculo de centroides
        for z in range(n_clust):
            index = [I for I,ind in enumerate(label) if ind==z]
            centroids[z] = np.mean(data[index], axis = 0)  
        # Criterio de convergencia: centroides reiterados o máximo de iteraciones alcanzado
        if ((centroids == seeds).all() or i == max_iter-1) : break  
        seeds = centroids # Actualización de centroides
        label = []
    
    return centroids, label
    

### Data experimental ###

data_1000 = rand(1000,2).astype(np.float64) # 1000 items
data_2000 = rand(2000,2).astype(np.float64) # 2000 items
data_3000 = rand(3000,2).astype(np.float64) # 3000 items
data_4000 = rand(4000,2).astype(np.float64) # 4000 items

k = 4 # Número de clústers a calcular
seeds = rand(k,2).astype(np.float64) # Centroides iniciales

### Listas de 1 elemento para map ###
d1 = [data_1000]
d2 = [data_2000]
d3 = [data_3000]
d4 = [data_4000]
s = [seeds]

if __name__ == '__main__': 
    t0 = tm.time()
    p = Pool(3)
    hola = p.map(kmeans, d1, s)
    t1 = tm.time()

    print(t1-t0)
'''
print(list(map(kmeans, [data_1000], [k], [seeds], [100])))


time_matrix = np.zeros((4,4)) # Matriz de tiempos de cálculos

for n_workers in range(1,5): # Ciclo que contempla proceso natural y multiprocessing
    if n_workers==1: # Procesamiento natural
        t0 = tm.time()
        kmeans(data_1000, k, seeds, 100)
        t1 = tm.time()
        kmeans(data_2000, k, seeds, 100)
        t2 = tm.time()
        kmeans(data_3000, k, seeds, 100)
        t3 = tm.time()
        kmeans(data_4000, k, seeds, 100)
        t4 = tm.time()
        time_matrix[0,0] = t1-t0
        time_matrix[1,0] = t2-t1
        time_matrix[2,0] = t3-t2
        time_matrix[3,0] = t4-t3
    else: # Multiprocessing de 2 hasta 4 workers
        if __name__ == '__main__': 
            

            t0 = tm.time()
            p = Pool(n_workers)
            p.map(kmeans, d1, [k], s, [100])
            t1 = tm.time()
            p = Pool(n_workers)
            p.map(kmeans, d2, [k], s, [100])
            t2 = tm.time()
            p = Pool(n_workers)
            p.map(kmeans, d3, [k], s, [100])
            t3 = tm.time()
            p = Pool(n_workers)
            p.map(kmeans, d4, [k], s, [100])
            t4 = tm.time()
            time_matrix[0,n_workers-1] = t1-t0
            time_matrix[1,n_workers-1] = t2-t1
            time_matrix[2,n_workers-1] = t3-t2
            time_matrix[3,n_workers-1] = t4-t3

print(time_matrix)
 


    
        
        

            



'''
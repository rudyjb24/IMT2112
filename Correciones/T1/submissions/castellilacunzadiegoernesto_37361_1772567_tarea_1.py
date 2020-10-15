import time
from multiprocessing import Pool, freeze_support

import matplotlib.pyplot as plt
import numpy as np


# Comienzo definiendo las funciones que voy a utilizar

def d_p(z):
    x = z[:2]
    y = z[2:]
    return (x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2


def d_l(x, y):
    return (x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2


def clustering(z):
    dato = z[0]
    centers = z[1]
    comparison = []
    for i in range(len(centers)):
        comparison.append(d_l(dato, centers[i]))
    comparison = np.array(comparison)
    return np.argmin(comparison)


def cluster_builder(data, index):
    clusters = []
    for i in range(n_clusters):
        cluster_i = []
        for j in range(len(index)):
            if index[j] == i:
                cluster_i.append(data[j])
        clusters.append(cluster_i)
    return clusters


def mass_center(lista):
    centers = n_clusters * [np.zeros(2)]
    centers_len = n_clusters * [0]
    for tupla in lista:
        dato = tupla[0]
        indice_cluster = tupla[1]
        centers[indice_cluster] = centers[indice_cluster] + dato
        centers_len[indice_cluster] += 1
    for i in range(len(centers)):
        if centers_len[i] != 0 or list(centers[i]) != list(np.zeros(2)):
            centers[i] *= (1/centers_len[i])
    return centers


# Apartir de aqui defino las funciones k_mean (Hago esto más que nada para mantenerme ordenado y no perderme :) )


def k_means_loops(data, centroid):
    centers = centroid
    clusters_index = []
    clusters_index_prev = [1]
    while clusters_index != clusters_index_prev:
        clusters_index = []
        for i in range(len(data)):
            x = data[i]
            distances = []
            for y in centers:
                distances.append(d_l(x, y))
            a = min(distances)
            index = distances.index(a)
            clusters_index.append(index)
        if clusters_index_prev != clusters_index:
            clusters_index_prev = clusters_index
            clusters_index = []
        centers = []
        for i in range(n_clusters):
            center_i = np.zeros(2)
            n_i = 0
            for j in range(len(clusters_index_prev)):
                if clusters_index_prev[j] == i:
                    center_i += data[j]
                    n_i += 1
            center_i *= (1 / n_i)
            centers.append(center_i)
    return clusters_index_prev


def k_means_numpy(data, centroid):
    centers = centroid
    clusters_index = []
    clusters_index_prev = [1]
    while clusters_index != clusters_index_prev:
        clusters_index = []
        for y in centers:
            array = data - y
            clusters_index.append(np.linalg.norm(array, axis=1))
        clusters_index = list(np.argmin(np.array(clusters_index), axis=0))
        if clusters_index_prev != clusters_index:
            clusters_index_prev = clusters_index
            clusters_index = []
        centers = []
        for q in range(n_clusters):
            index_array = np.array(clusters_index_prev)
            cluster_size = np.sum(index_array == q)
            where = np.array([index_array == q, index_array == q]).T
            centers.append((1 / cluster_size) * np.sum(data, where=where, axis=0))
    return clusters_index_prev


def k_means_multiproccessing(data, centroid):
    centers = centroid
    clusters_index = []
    clusters_index_prev = [1]
    while clusters_index != clusters_index_prev:
        data = list(data)
        centers_conect = len(data) * [centers]
        proces_data = list(zip(data, centers_conect))
        p = Pool(n_workers)
        clusters_index = list(p.map(clustering, proces_data))
        p.close()
        p.join()
        freeze_support()
        if clusters_index_prev != clusters_index:
            clusters_index_prev = clusters_index
            clusters_index = []
        index_data = list(zip(data, clusters_index_prev))
        chuncks = []
        size_chunks = int(len(data)/n_workers)
        for i in range(n_workers-1):
            chuncks.append(index_data[i*size_chunks:(i+1)*size_chunks])
        chuncks.append(index_data[(n_workers-1)*size_chunks:])
        p = Pool(n_workers)
        centers_chunck = list(p.map(mass_center, chuncks))
        p.close()
        p.join()
        freeze_support()
        centers = np.array(centers_chunck[0])
        for i in range(n_workers-1):
            vector = np.array(centers_chunck[i+1])
            centers += vector
        centers *= (1/n_workers)
        centers = list(centers)
    return clusters_index_prev


# Ahora defino constantes para controlar el ejercicio

n_workers = 4
datos = np.random.randn(10000, 2)
n_clusters = 10
centros = np.random.randn(n_clusters, 2)
centros = list(centros)

# Ahora corro el codigo (Windows :S)

if __name__ == '__main__':
    t0 = time.time()
    loop_index = k_means_loops(datos, centros)
    t1 = time.time()
    print('Tiempo de demora de Loops es: ', t1 - t0)
    clusters_loop = cluster_builder(datos, loop_index)
    for h in range(len(clusters_loop)):
        cluster_loop = np.array(clusters_loop[h])
        X = cluster_loop.T[0]
        Y = cluster_loop.T[1]
        number = 'Cluster ' + str(h)
        plt.scatter(X, Y, label=number)
    plt.show()

    t0 = time.time()
    numpy_index = k_means_numpy(datos, centros)
    t1 = time.time()
    print('Tiempo de demora de NumPy es: ', t1 - t0)
    clusters_numpy = cluster_builder(datos, numpy_index)
    for h in range(len(clusters_numpy)):
        cluster_np = np.array(clusters_numpy[h])
        X = cluster_np.T[0]
        Y = cluster_np.T[1]
        number = 'Cluster ' + str(h)
        plt.scatter(X, Y, label=number)
    plt.show()

    t0 = time.time()
    clusters_multiproccess = k_means_multiproccessing(datos, centros)
    t1 = time.time()
    print('Tiempo de demora de MultiProccessing es: ', t1 - t0)
    for h in range(len(clusters_multiproccess)):
        cluster_mp = np.array(clusters_multiproccess[h])
        X = cluster_mp.T[0]
        Y = cluster_mp.T[1]
        number = 'Cluster ' + str(h)
        plt.scatter(X, Y, label=number)
    plt.show()

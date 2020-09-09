import numpy as np
import random as rd
from multiprocessing import Pool
import math
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import time as tm
import matplotlib.pyplot as plt

#1.

X, y = make_blobs(n_samples=1000,n_features=2) #1000 samples de 2 dimensiones con valores entre -10 y 10

#2.

def kmeansklearn(k,X):
    kmeans = KMeans(n_clusters=k).fit(X)
    return kmeans.cluster_centers_

#3. a)

def kmeanslist(k,X):
    C = [list(rd.choice(X)) for j in range(k)]
    for i in range(300): #tal y como hace el algoritmo k-means de sklearn, pondremos un maximo de 300 iteraciones
        labels = [] #lista de labels
        for x in X: #asignar el numero de cluster a cada sample
            dC = []
            for c in C:
                d = 0
                for j in range(2):
                    d = (x[j]-c[j])**2
                dC.append(math.sqrt(d))
            labels.append(dC.index(min(dC)))
        for l in range(k): #calculamos los centros de masa decada cluster y los asignamos como centros
            C[l][0] = 0
            C[l][1] = 0
            for j in range(X.shape[0]):
                if labels[j] == l:
                    C[l][0] += X[j][0]
                    C[l][1] += X[j][1]
            C[l][0] = C[l][0]/X.shape[0]
            C[l][1] = C[l][1]/X.shape[0]
    return C

#b)

def kmeansnumpy(k,X):
    C = np.array([list(rd.choice(X)) for j in range(k)]) #creamos los centros elegidos aleatoriamente de entre los samples
    for i in range(300):
        labels = np.array([])
        for x in X: #asignar cada sample a un cluster
            dC = np.array([])
            for c in C:
                dC = np.append(dC,np.linalg.norm(x-c))
            labels = np.append(labels,np.argmin(dC))
        for l in range(k): #calculamos los centros de masa decada cluster y los asignamos como centros
            C[l] = np.average(X[labels == l],axis=0)
    return C

#c)

def coordenadacentro(X): #toma algun dato aleatorio como centro
    return list(rd.choice(X))

k=4
C0 = 0
centros  = 0
if __name__ == '__main__':
    p = Pool(8)
    centros = p.map(coordenadacentro,[X for i in range(k)])
C0 = np.array(centros)

def distanciaalcentro(x,c):
    return np.linalg.norm(x - c)

def centromin(x,C):  # encuentra el centro mas cercano a x
    d = 0
    if __name__ == '__main__':
        p = Pool(8)
        dC = p.starmap(distanciaalcentro,[(x,c) for c in C])
        d = dC
        p.close()
        p.join()
    return np.argmin(np.array(d))

def asignar_labels(X,C):  # crea un array de labels de los centros mas cercanos
    l = 0
    if __name__ == '__main__':
        p = Pool(8)
        labels = list(p.starmap(centromin,[(x,C) for x in X]))
        l = labels
        p.close()
        p.join()
    return np.array(l)

def cdm(labels,X,i):  # entrega el centro de masa
    return np.average(X[labels == i], axis=0)

def centros(labels, C, X):  # asigna los nuevos centros
    j = 0
    if __name__ == '__main__':
        p = Pool(8)
        centros = p.starmap(cdm,[(labels,X,i) for i in range(C.shape[0])])
        j = centros
        p.close()
        p.join()
    return np.array(j)

def kmeansparalelo(C,X):  # el algoritmo de kmeans con multiproccessing que entrega los centros
    for i in range(300):
        labels = asignar_labels(X,C)
        C = centros(labels,C,X)
    return C

#4. Ocupamos los 4 metodos para 1000, 2000 y 5000 datos (todos con k = 4). Comenzamos con 1000.

t1 = tm.time()
kmeanslist(4,X)
t2 = tm.time()
print("Tiempo solo con Loops y 1000 datos: {}\n".format(round(t2-t1, 6)))

t1 = tm.time()
kmeanslist(4,X)
t2 = tm.time()
print("Tiempo con Numpy y 1000 datos: {}\n".format(round(t2-t1, 6)))

t1 = tm.time()
kmeanslist(4,X)
t2 = tm.time()
print("Tiempo con multiprocessing y 1000 datos: {}\n".format(round(t2-t1, 6)))

t1 = tm.time()
kmeansklearn(4,X)
t2 = tm.time()
print("Tiempo con sklearn.KMeans y 1000 datos: {}\n".format(round(t2-t1, 6)))

#Ahora con 2000

X, y = make_blobs(n_samples=2000,n_features=2)

t1 = tm.time()
kmeanslist(4,X)
t2 = tm.time()
print("Tiempo solo con Loops y 2000 datos: {}\n".format(round(t2-t1, 6)))

t1 = tm.time()
kmeanslist(4,X)
t2 = tm.time()
print("Tiempo con Numpy y 2000 datos: {}\n".format(round(t2-t1, 6)))

t1 = tm.time()
kmeanslist(4,X)
t2 = tm.time()
print("Tiempo con multiprocessing y 2000 datos: {}\n".format(round(t2-t1, 6)))

t1 = tm.time()
kmeansklearn(4,X)
t2 = tm.time()
print("Tiempo con sklearn.KMeans y 2000 datos: {}\n".format(round(t2-t1, 6)))

#Ahora con 5000

X, y = make_blobs(n_samples=5000,n_features=2)

t1 = tm.time()
kmeanslist(4,X)
t2 = tm.time()
print("Tiempo solo con Loops y 5000 datos: {}\n".format(round(t2-t1, 6)))

t1 = tm.time()
kmeanslist(4,X)
t2 = tm.time()
print("Tiempo con Numpy y 5000 datos: {}\n".format(round(t2-t1, 6)))

t1 = tm.time()
kmeanslist(4,X)
t2 = tm.time()
print("Tiempo con multiprocessing y 5000 datos: {}\n".format(round(t2-t1, 6)))

t1 = tm.time()
kmeansklearn(4,X)
t2 = tm.time()
print("Tiempo con sklearn.KMeans y 5000 datos: {}\n".format(round(t2-t1, 6)))

#5. Graficamos 1000 datos

X, y = make_blobs(n_samples=1000,n_features=2)

C = kmeanslist(4,X)
labels = asignar_labels(X,C)

plt.subplot(221)
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.title("Loops")

C = kmeansnumpy(4,X)
labels = asignar_labels(X,C)

plt.subplot(221)
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.title("Numpy")


k=4
c0 = 0
cen = 0
if __name__ == '__main__':
    p = Pool(8)
    cen = p.map(coordenadacentro,[X for i in range(k)])
c0 = np.array(cen)
C = kmeansparalelo(c0,X)
labels = asignar_labels(X,C)

plt.subplot(221)
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.title("multiprocessing")


labels = KMeans(n_clusters=4).fit_predict(X)

plt.subplot(224)
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.title("sklearn.KMeans")

plt.show()
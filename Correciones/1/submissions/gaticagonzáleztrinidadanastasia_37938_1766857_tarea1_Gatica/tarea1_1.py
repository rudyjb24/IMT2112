import numpy as np
import matplotlib.pyplot as plt 
from time import time

start = time()
#n datos de dimensión m 
n = 10000
m = 2
data = np.random.rand(n, m)

def menor(lista, cent, k):
    l = []
    for p in range(k):
        for i in range(len(lista)):
            if lista[i] == cent[p][i]:
                l.append(p)
    return l

def media(lista):
    cantidad = len(lista)
    l = np.array(lista)
    if cantidad != 0:
        med = sum(l)/cantidad
        return med
    else: 
        return sum(l)

def centroides(k, data, centros, iter):
    if iter < 1:
        centroide = np.random.rand(k, m)
        cent = []
        for i in range(k):
            cent.append(np.array(list(map(lambda x: np.linalg.norm(centroide[i]- x), data))))
        menores = np.array(list(map(lambda x, y, z: min(x, y, z) ,cent[0], cent[1], cent[2])))
        centros = menor(menores, cent, k)
        
    else: 
        centro = []
        for j in range(k):
            lista = []
            for i in centros:
                if i == j:
                    lista.append(i)
            centro.append(media(lista))
        centroide = np.array(centro)
        cent = []
        for i in range(k):
            cent.append(np.array(list(map(lambda x: np.linalg.norm(centroide[i]- x), data))))
        menores = np.array(list(map(lambda x, y, z: min(x, y, z) ,cent[0], cent[1], cent[2])))
        centros = menor(menores, cent, k)
    
    if sum(menores) < n/2:
        return [data, centroide]
 
        
    else: 
        iter += 1
        centroides(k, data ,centros, iter)

centros = []
resultado = centroides(3, data, centros, 0)
total = time() - start
print(total)


#descomprimimos el resultado
data = resultado[0]
centroide = resultado[1]


###Ploteamos 
for i in range(len(data)):
    plt.plot(data[i][0], data[i][1], "*")
for j in range(len(centroide)):
    plt.plot(centroide[j][0],centroide[j][1], "+")
plt.show()
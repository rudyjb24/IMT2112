import numpy as np
import matplotlib.pyplot as plt
from time import time


start = time()
##Funciones a utilizar
def menor(cent, k, n):
    l = []
    for i in range(n):
        men = 2
        for j in range(k):
            if cent[j][i] < men:
                men = cent[j][i]
        l.append(men)
    return l

def minimo(cent, l, n, k):
    lista = []
    for j in range(k):
        for i in range(n):
            if l[i] == cent[j][i]:
                lista.append(j)
    return lista

def media(lista):
    cantidad = len(lista)
    l = np.array(lista)
    if cantidad != 0:
        med = sum(l)/cantidad
        return med
    else: 
        return sum(l)

#Generar n registros de m dimensiones cada uno.
m = 2
n = 100

data = np.random.rand(n, m)



#Inicializar el algoritmo.
def centroides(k, data, centros, iter):
    if iter < 1:
        centroide = np.random.rand(k, m)
        cent = []
        for i in range(k):
            l = []
            for j in range(n):
                l.append(sum(abs(centroide[i]- data[j])))
            cent.append(l)
        menores = menor(cent, k, n)
        centros = minimo(cent, menores, n, k)
    
    #tomar el centro de masa de cada cluster y asignar los centroides a esos datos
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
            l = []
            for j in range(n):
                l.append(sum(abs(centroide[i]- data[j])))
            cent.append(l)
        menores = menor(cent, k, n)
        centros = minimo(cent, menores, n, k)



    
    
    suma = 0
    for i in range(n):
        suma += menores[i]
    if suma < n/2:
        return [data, centroide]
    
    #iterar sobre los centroides  
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
import numpy as np
from time import time
def norma(dato,centroide):
    """Norma L1 entre dos datos distintos"""
    valor = 0
    for i in range(len(dato)):
        valor +=  (dato[i]-centroide[i])**2
    return valor**0.5

def suma(datos):
    lista = []
    for dim in range(len(datos[0])):
        suma = 0
        for dato in datos:
            suma += dato[dim]
        lista.append(suma/len(datos))
    return lista


def k_mean(k,data,centriods = None, tol = 0):
    """Algoritmo que k-mean con k centroides para los datos data"""

    # Se generan k centroides
    if type(centriods) != type(None):
        centroides = centriods
    else:
        # Unico momento que se ocupa numpy es para crear
        # Centroides aleatorios.
        centroides = np.random.rand(k,len(data[0]))

    lista_datos = []
    for i in range(k):
        lista_datos.append([centroides[i],[]])

    # dist < maxima_distancia para L1
    maxima_distancia = len(data) + 1

    # se calcula la distancia de los datos a los centriodes y
    # se asigna un centroide para cada dato.
    for dato in data:
        minimo = maxima_distancia
        for centroide in range(k):
            posible_min = min(minimo, norma(dato,centroides[centroide]))
            if posible_min < minimo:
                c_temp = centroide
                minimo = posible_min

        lista_datos[c_temp][1].append(dato)

    # se calculan los nuevos centriodes.
    centroides_temp = []
    for centroide in range(k):
        # Si no se le asigna ningún punto al centroide, este se ignora
        if len(lista_datos[centroide][1]) != 0:
            c_temp = np.array(suma(lista_datos[centroide][1]))
        # En otro caso se actualizan las coordenadas
        else:
            c_temp=lista_datos[centroide][0]
        centroides_temp.append(c_temp)
        lista_datos[centroide][0] = c_temp

    # si los centriodes no se desplazan se finaliza el algoritmo
    for indice in range(len(centroides_temp)):
        # len(centroides[0]) = dimension en la que trabajamos
        for dim in range(len(centroides[0])):
            # si los centroides se desplazan más que cierta tolerancia, se itera
            if abs(centroides_temp[indice][dim]-centroides[indice][dim])> tol:
                return k_mean(k,data,centriods = centroides_temp, tol = tol)
    # si la los centros se mantienen dentro de cierta tolerancia se finaliza
    return lista_datos

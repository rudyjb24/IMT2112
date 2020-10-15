import numpy as np
from numpy.linalg import norm

def k_mean(k,data,centriods = None, tol = 0):
    """Algoritmo que k-mean con k centroides para los datos data"""

    # Se generan k centroides
    if type(centriods) != type(None):
        centroides = centriods
    else:
        centroides = np.random.rand(k,len(data[0]))

    lista_datos = []
    for i in range(k):
        lista_datos.append([centroides[i],[]])

    # dist < maxima_distancia para L1 > L2
    maxima_distancia = len(data) + 1

    # se calcula la distancia de los datos a los centriodes y
    # se asigna un centroide para cada dato.

    for dato in data:
        minimo = maxima_distancia
        for centroide in range(k):
            posible_min = min(minimo, norm(dato-centroides[centroide],2))
            if posible_min < minimo:
                c_temp = centroide
                minimo = posible_min

        lista_datos[c_temp][1].append(dato)

    # se calculan los nuevos centriodes.
    centroides_temp = []
    for centroide in range(k):
        # Si  se le asigna ningún punto al centroide, este se actualiza
        if len(lista_datos[centroide][1]) != 0:
            c_temp = sum(lista_datos[centroide][1])/len(lista_datos[centroide][1])
            #el agoritmo de suma es vectorial
        # En otro caso  este se ignora
        else:
            c_temp=lista_datos[centroide][0]
        centroides_temp.append(c_temp)
        lista_datos[centroide][0] = c_temp

    # si los centriodes no se desplazan se finaliza el algoritmo

    if np.any(np.any(abs(np.array(centroides_temp)-np.array(centroides))> tol)):
        return k_mean(k,data,centriods = centroides_temp, tol = tol)

    # si la los centros se mantienen dentro de cierta tolerancia se finaliza
    return lista_datos

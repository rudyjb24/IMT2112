import multiprocess as mp
import numpy as np
from numpy.linalg import norm
from time import time as t



# Pool
n_workers = 4

def asignacion(arg):
    data, centroides = arg[0],arg[1]
    k = len(centroides)
    maxima_distancia = len(data) + 1
    lista_datos = []
    for i in range(k):
        lista_datos.append([])
    for dato in data:
        minimo = maxima_distancia
        for centroide in range(k):
            posible_min = min(minimo, norm(dato-centroides[centroide],2))
            if posible_min < minimo:
                c_temp = centroide
                minimo = posible_min
        lista_datos[c_temp].append(dato)
    return lista_datos


def k_mean(k,data,centriods = None, tol = 0,workers = None,n_w= n_workers):
    """Algoritmo que k-mean con k centroides para los datos data"""
    global lista_datos
    # Se generan k centroides
    if type(centriods) != type(None):
        centroides = centriods
    else:
        centroides = np.random.rand(k,len(data[0]))

    # dist < maxima_distancia para L1 > L2
    maxima_distancia = len(data) + 1

    # se divide la data para cada chuncks
    size_chuncks = int(len(data)/n_workers)
    chuncks = []

    for i in range(n_workers-1):
        chuncks.append([data[i*size_chuncks:(i+1)*size_chuncks],centroides])

    chuncks.append([data[(i+1)*size_chuncks:],centroides])

    # se calcula la distancia de los datos a los centroodes y
    # se asigna un centroide para cada dato.
    temp_1 = workers.map(asignacion,chuncks)

    lista_datos = []
    for i in range(k): ## cantidad de centroides
        temp_2 = []
        for j in range(n_w): ## catidad de n_trabajadores
            temp_2.extend(temp_1[j][i])
        lista_datos.append([centroides[i],temp_2])

    # se calculan los nuevos centroides.
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

    # si los centroides no se desplazan se finaliza el algoritmo

    if np.any(np.any(abs(np.array(centroides_temp)-np.array(centroides))> tol)):
        return k_mean(k,data,centriods = centroides_temp, tol = tol,workers = workers,n_w= n_workers)

    # si la los centros se mantienen dentro de cierta tolerancia se finaliza
    return lista_datos



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    cantidad_datos = 10000
    colores = ["r","g","b","cyan","gray","black","yellow"]
    data = np.random.rand(cantidad_datos,2)
    k = 4
    w = mp.Pool(n_workers)
    import time

    t0 = time.time()
    means = k_mean(k,data,workers=w)
    print(time.time()-t0)
    for i in range(k):
        for dato in means[i][1]:
            plt.plot(dato[0],dato[1],"*",c=colores[i])
    for i in range(k):
        plt.plot(float(means[i][0][0]),float(means[i][0][1]),"*",c="black")
    plt.title(f"K-means con {k} centros y {cantidad_datos} datos\n"
                f"Usando k_mean_multiprocess (y {n_workers} n_trabajadores)")
    plt.show()

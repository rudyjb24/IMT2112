import numpy as np
from multiprocessing import Pool
import time
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

#generamos los datos: n registros de m dimensiones, reescalados en r

n = 5000
m = 7
r = 1

#numero de centros
k = 5

registros = []

#Generamos los registros con k ruidos distintos (el k-ésimo es [0,0,0,0,0,0,0]), para intentar que se noten los k clusters
#en el grafico luego de la reduccion de dimensionalidad. Este ruido tambien causa que el algoritmo sea más rápido.
#En los tiempos reportados hay reportes con ruidos y sin ruidos.
registros = []
for i in range(n):
    if i < n/5:
        registros.append(r*np.random.rand(m) + np.array([1,1,0,0,0,0,0]))
    elif n/5 <= i <2*n/5:
        registros.append(r*np.random.rand(m) + np.array([-1,-1,0,0,0,0,0]))
    elif 2*n/5 <= i <3*n/5:
        registros.append(r*np.random.rand(m) + np.array([1,-1,0,0,0,0,0]))
    elif 3*n/5 <= i <4*n/5:
        registros.append(r*np.random.rand(m) + np.array([-1,1,0,0,0,0,0]))
    else:
        registros.append(r*np.random.rand(m))

#descomentar para generar registros random, sin ruido
"""registros = []
for i in range(n):
    registros.append(r*np.random.rand(m))"""

#Primero programamos k-means usando loops
def k_means_loops(r, n, registros, m, k):

    #elegimos los centros iniciales como los primeros k registros
    centros_iniciales = []
    for i in range(k):
        centros_iniciales.append(registros[i])


    #etiquetas[i] guarda el indice del centro mas cercanos a registros[i]
    etiquetas_iniciales = []

    #calculamos la etiquetas iniciales para cada registro
    count_registro = 0
    for registro in registros:
        #distancia maxima
        distancia_definitiva = 4*r + 1
        #numero de centro
        count_centros = 0
        #etiqueta
        etiqueta = 0
        for centro in centros_iniciales:
            #variable que almacena distancia al centro 'centro'
            dist_parcial = 0

            #calculamos la distancia euclideana al centro 'centro'
            for coordenada in range(m):
                dist_parcial += (registro[coordenada]-centro[coordenada])**2

            dist_parcial = dist_parcial**(1/2)
            #vemos si está mas cerca que los anteriores
            if dist_parcial < distancia_definitiva:
                distancia_definitiva = dist_parcial
                etiqueta = count_centros

            #actualizamos el contador
            count_centros += 1
        #asignamos el centro
        etiquetas_iniciales.append(etiqueta)
        count_registro += 1

    #Que los clusters no cambien significa que no cambian las etiquetas ni los centros de masa de los clusters

    #Guardaremos las etiquetas despues del loop para ver que se hayan fijado las etiquetas:
    etiquetas_finales = []
    for _ in registros:
        etiquetas_finales.append(-1)

    #Guardaremos los centros de masa despues del loop para ver que se hayan fijado:
    centros_finales = []
    for i in range(k):
        centros_finales.append(np.zeros(m))

    #while hasta que los clusters se fijen
    iteraciones = 1
    while True:

        #Cantidad de registros asignados a cada centro de masa, parten con 0
        cantidad_etiquetas = dict()
        for i in range(k):
            cantidad_etiquetas[i] = 0

        #recorremos todos los registros y le asignamos su centro de masa mas cercano
        count_registro = 0
        for registro in registros:
            #distancia maxima
            distancia_definitiva = 4*r + 1
            #numero de centro
            count_centros = 0
            #etiqueta
            etiqueta = 0

            for centro in centros_iniciales:

                #variable que almacena distancia al centro 'centro'
                dist_parcial = 0

                #calculamos la distancia euclideana al centro 'centro'
                for coordenada in range(m):
                    dist_parcial += (registro[coordenada]-centro[coordenada])**2

                dist_parcial = dist_parcial**(1/2)

                #vemos si está mas cerca que los anteriores
                if dist_parcial < distancia_definitiva:
                    distancia_definitiva = dist_parcial
                    etiqueta = count_centros

                #actualizamos el contador
                count_centros += 1

            #asignamos el centro
            etiquetas_finales[count_registro] = etiqueta

            #sumamos la etiqueta al contador de etiquetas:
            cantidad_etiquetas[etiqueta] += 1
            count_registro += 1

        #calculamos los centros finales con las nuevas etiquetas
        #recorremos los registros
        for i in range(n):
            #recorremos sus coordenadas
            for coordenada in range(m):
                centros_finales[etiquetas_finales[i]][coordenada] += registros[i][coordenada]/cantidad_etiquetas[etiquetas_finales[i]]

        #vemos si los centros ya se fijaron
        centros_terminados = True
        for i in range(k):
            #si hay alguna coordenada distinta entre los centros finales y los iniciales, no se han fijado los centros
            for coordenada in range(m):
                if centros_iniciales[i][coordenada] != centros_finales[i][coordenada]:
                    centros_terminados = False

        #vemos si se fijaron las etiquetas
        etiquetas_terminadas = True
        for i in range(n):
            if etiquetas_iniciales[i] != etiquetas_finales[i]:
                etiquetas_terminadas = False

        #vemos si se fijaron los clusters
        if etiquetas_terminadas and centros_terminados:
            return etiquetas_iniciales, centros_iniciales

        #actualizamos los centros_finales
        for i in range(k):
            centros_iniciales[i] = centros_finales[i]
            centros_finales[i] = np.zeros(m)

        #actualizamos las etiquetas:
        for i in range(n):
            etiquetas_iniciales[i] = etiquetas_finales[i]
            etiquetas_finales[i] = -1

        #pasamos a la siguiente iteración
        iteraciones += 1

#Funcion con vectores.
#La lógica es exactamente la misma que en la función anterior así que no se comentará de nuevo
def k_means_vectorizado(r, n, registros, m, k):

    centros_iniciales = np.array([registros[i] for i in range(k)])
    etiquetas_iniciales = np.array([None for _ in registros])

    for j in range(n):
        distancia_definitiva = 4*r + 1
        etiqueta = 0

        for i in range(k):
            dist = np.linalg.norm(registros[j]-centros_iniciales[i])
            if dist < distancia_definitiva:
                distancia_definitiva = dist
                etiqueta = i
        etiquetas_iniciales[j] = etiqueta

    etiquetas_finales = -np.ones(n, dtype = int)
    centros_finales = np.zeros((k,m))

    iteraciones = 1
    while True:

        cantidad_etiquetas = {i:0 for i in range(k)}

        for j in range(n):
            distancia_definitiva = 4*r + 1
            etiqueta = 0

            for i in range(k):
                dist_parcial = np.linalg.norm(registros[j]-centros_iniciales[i])
                if dist_parcial < distancia_definitiva:
                    distancia_definitiva = dist_parcial
                    etiqueta = i

            etiquetas_finales[j] = etiqueta
            cantidad_etiquetas[etiqueta] += 1

        for i in range(n):
            centros_finales[etiquetas_finales[i]]+= registros[i]/cantidad_etiquetas[etiquetas_finales[i]]

        if (centros_iniciales == centros_finales).all() and (etiquetas_iniciales == etiquetas_finales).all():
            return etiquetas_iniciales, centros_iniciales

        centros_iniciales = centros_finales
        centros_finales = np.zeros((k,m))
        etiquetas_iniciales = etiquetas_finales
        etiquetas_finales = -np.ones(n, dtype = int)
        iteraciones += 1

#Aquí comienza la parte paralela

#numero de trabajadores
n_trabajadores = 4

#Funciones que se usaran en el código paralelo
def crear_etiquetas_iniciales(x):
    return None

def limpiar_etiquetas_finales(x):
    return -1

#paralelizaremos el loop sobre los registros que calcula la etiqueta asignada a cada registro
def loop_registros(inp):

    registro = inp[0]
    centros = inp[1]
    #distancia maxima
    distancia_definitiva = 4*r + 1
    #numero de centro
    count_centros = 0
    #etiqueta
    etiqueta = 0
    for centro in centros:
        #variable que almacena distancia al centro 'centro'
        dist_parcial = 0

        #calculamos la distancia euclideana al centro 'centro'
        for coordenada in range(m):
            dist_parcial += (registro[coordenada]-centro[coordenada])**2

        dist_parcial = dist_parcial**(1/2)
        #vemos si está mas cerca que los anteriores
        if dist_parcial < distancia_definitiva:
            distancia_definitiva = dist_parcial
            etiqueta = count_centros

        count_centros += 1

    return etiqueta

#esto es para paralelizar el calculo de centros finales. Notamos que solo se hacen m sumas por cada registro, por lo tanto
#es más eficiente no paralelizar esta parte. En la practica no lo hice, pero dejé comentada su paralelización de todas formas

def centro_final(lista_registros):

    centro_final = np.zeros(m)
    for registro in lista_registros:
        for coordenada in range(m):
            centro_final[coordenada] += registro[coordenada]/len(lista_registros)

    return centro_final

#Funcion k-means paralela. Es la misma lógica que los otros dos, así que no se comentará mucho tampoco.
def k_means_paralelo(r, n, registros, m, k):

    centros_iniciales = []
    for i in range(k):
        centros_iniciales.append(registros[i])

    with Pool(processes = n_trabajadores) as p:
        etiquetas_iniciales = list(p.map(loop_registros, [(registros[i], centros_iniciales) for i in range(n)]))
        etiquetas_finales = list(p.map(limpiar_etiquetas_finales, registros))

    centros_finales = []
    for i in range(k):
        centros_finales.append(np.zeros(m))

    #while hasta que los centros no cambien
    iteraciones = 1
    while True:

        #cuenta cuantos registros se le asignan a un centro
        cantidad_etiquetas = dict()
        for i in range(k):
            cantidad_etiquetas[i] = 0

        #recorremos todos los registros y le asignamos su centro mas cercano
        with Pool(processes = n_trabajadores) as p:
            etiquetas_finales = list(p.map(loop_registros, [(registros[i], centros_iniciales) for i in range(n)]))

            #sumamos la etiqueta al contador de etiquetas:
        for etiqueta in etiquetas_finales:
            cantidad_etiquetas[etiqueta] += 1


        #calculamos los centros finales con las nuevas etiquetas
        #recorremos los registros
        for i in range(n):
            #recorremos sus coordenadas
            for coordenada in range(m):
                centros_finales[etiquetas_finales[i]][coordenada] += registros[i][coordenada]/cantidad_etiquetas[etiquetas_finales[i]]

        """
        EL FOR ANTERIOR PARALELIZADO HUBIERA SIDO ASÍ:

        chuncks = [[] for _ in range(n)]
        for i in range(n):
            chuncks[etiquetas_finales[i]].append(registros[i])

        with Pool(processes = k) as p:
            centros_finales = list(p.map(centro_final, chuncks))

        PERO ERA MÁS LENTO, POR LO EXPLICADO MÁS ARRIBA CUANDO SE DEFINE LA FUNCIÓN centro_final.

        """

        #No hay mas loops sobre los registros asi que de aqui en adelante es similiar a la funcion con loops

        centros_terminados = True
        for i in range(k):

            for coordenada in range(m):
                if centros_iniciales[i][coordenada] != centros_finales[i][coordenada]:
                    centros_terminados = False

        etiquetas_terminadas = True
        for i in range(n):
            if etiquetas_iniciales[i] != etiquetas_finales[i]:
                etiquetas_terminadas = False

        if etiquetas_terminadas and centros_terminados:
            return etiquetas_iniciales, centros_iniciales

        for i in range(k):
            centros_iniciales[i] = centros_finales[i]
            centros_finales[i] = np.zeros(m)

        for i in range(n):
            etiquetas_iniciales[i] = etiquetas_finales[i]
            etiquetas_finales[i] = -1

        iteraciones += 1


#Aquí se ejecutan los 3 algortimos y se toman los tiempos
t0 = time.time()
l_et, l_cen = k_means_loops(r, n, registros, m, k)
t1 = time.time()
print('tiempo loops: ', t1-t0)
t0 = time.time()
v_et, v_cen = k_means_vectorizado(r, n, registros, m, k)
t1 = time.time()
print('tiempo vectores: ', t1-t0)
t0 = time.time()
p_et, p_cen = k_means_paralelo(r, n, registros, m, k)
t1 = time.time()
print('tiempo paralelos: ', t1-t0)
#print(l_et == p_et, (np.array(l_cen) == np.array(p_cen)).all()) esto comprueba que los outputs son iguales


#VISUALIZACIÓN para 5 centros (dato: mi PC se demora mucho en graficar para mas de 5.000 registros)
pca = PCA(n_components=2)
registros = pca.fit(registros).transform(registros)

for i in range(n):
    if v_et[i] == 0:
        plt.scatter(registros[i][0], registros[i][1], c = 'green')
    elif v_et[i] == 1:
        plt.scatter(registros[i][0], registros[i][1], c = 'red')
    elif v_et[i] == 2:
        plt.scatter(registros[i][0], registros[i][1], c = 'black')
    elif v_et[i] == 3:
        plt.scatter(registros[i][0], registros[i][1], c = 'brown')
    elif v_et[i] == 4:
        plt.scatter(registros[i][0], registros[i][1], c = 'yellow')
plt.show()

#TIEMPOS CON DATOS RANDOM

"""
Con
n = 10000
m = 7
r = 1
k = 5
los tiempos fueron
tiempo loops:  30.642690181732178
tiempo vectores:  28.28740692138672
tiempo paralelos:  24.608949899673462
"""

"""
Con
n = 15000
m = 7
r = 1
k = 5
los tiempos fueron
tiempo loops:  79.81594896316528
tiempo vectores:  73.41628098487854
tiempo paralelos:  66.37701106071472
"""

"""
Con
n = 20000
m = 7
r = 1
k = 5
los tiempos fueron
tiempo loops:  76.82133102416992
tiempo vectores:  71.27980184555054
tiempo paralelos:  60.89993596076965
"""

#TIEMPOS CON DATOS CON RUIDOS INICIALES PARA CLUSTERIZAR MAS FACIL Y QUE SE NOTEN LOS CLUSTERS LUEGO DE REDUCIR DIMENSION:
"""
Con
n = 5000
m = 7
r = 1
k = 5
los tiempos fueron
tiempo loops:  2.5699520111083984
tiempo vectores:  2.2933101654052734
tiempo paralelos:  2.3965110778808594
"""

"""
Con
n = 10000
m = 7
r = 1
k = 5
los tiempos fueron
tiempo loops:  14.885258674621582
tiempo vectores:  13.174176216125488
tiempo paralelos:  11.806832075119019
"""

"""
Con
n = 15000
m = 7
r = 1
k = 5
los tiempos fueron
tiempo loops:  9.902920961380005
tiempo vectores:  9.89261507987976
tiempo paralelos:  8.053147077560425
"""

"""
Con
n = 20000
m = 7
r = 1
k = 5
los tiempos fueron
tiempo loops:  26.46412420272827
tiempo vectores:  25.513909101486206
tiempo paralelos:  20.71623182296753
"""

"""
Con
n = 100000
m = 7
r = 1
k = 5
los tiempos fueron
tiempo loops:  52.796795129776
tiempo vectores:  50.40371489524841
tiempo paralelos:  41.35233187675476
"""


#HILOS Y PROCESOS
"""
Loops:

1 proceso
3 hilos

Vectorizado:

1 proceso
3 hilos

Paralelo:
Es variable la cantidad de procesos durante el desarrollo del algoritmo
Esta el proceso principal. Luego, cuando paraleliza, como tengo 4 trabajadores,
me salen a veces 4 procesos más con un hilo cada uno y en este momento el proceso principal
tiene 4 hilos, uno por proceso paralelo.

Pero a veces me salen 8 procesos paralelos, de un hilo todos. En este momento el proceso
princial que controla a los demás sigue teniendo 4 (n_trabajadores) hilos. Esto porque cada
hilo del proceso principal controla a dos procesos paralelos.

En resumen, todos los procesos paralelos tienen 1 hilo y el principal tiene 4 cuando hay
4 u 8 procesos paralelos. Y por ende, en total hay 5 o 9 procesos de python.
"""

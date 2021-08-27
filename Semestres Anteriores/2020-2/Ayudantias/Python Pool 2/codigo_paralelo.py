import multiprocessing
import time as tm

def calcular_promedio(notas):

    notas = notas.strip().split(',')
    notas = [float(x) for x in notas]
    PC = (notas[0] + notas[1] + 2*notas[2] - min(notas[:3]))/3
    PT = (notas[3] + notas[4] + notas[5] + notas[6] + notas[7] + notas[8] + notas[9] - min(notas[3:]))/6
    PF = round(0.7*PC + 0.3*PT, 2)

    if notas[2] >= 4:
        PF = max(PF, 4)

    return PF

def calcular_reprobados(notas_finales):

    reprobados = 0

    for nota in notas_finales:
        if nota < 4:
            reprobados += 1

    return reprobados


t_inicial = tm.time()

t0 = tm.time()
n_trabajadores = 4

with open("notas.txt", 'r') as file:
    file.readline()
    alumnos = file.readlines()
t1 = tm.time()
tiempo_secuencial = t1-t0
print('Tiempo de lectura de datos: ', t1-t0)

t0 = tm.time()
p = multiprocessing.Pool(n_trabajadores)
t1 = tm.time()
tiempo_secuencial += t1-t0
print('Tiempo de creacion del pool: ', t1-t0)

t0 = tm.time()
notas_finales = list(p.map(calcular_promedio, alumnos))
t1 = tm.time()
tiempo_paralelo = t1 - t0
print('Tiempo calculo promedios: ', t1-t0)

t0 = tm.time()
cantidad_alumnos = len(notas_finales)

size_chuncks = int(cantidad_alumnos/n_trabajadores)
chuncks = []

for i in range(n_trabajadores-1):
    chuncks.append(notas_finales[i*size_chuncks:(i+1)*size_chuncks])

chuncks.append(notas_finales[(i+1)*size_chuncks:])

t1 = tm.time()
tiempo_secuencial += t1-t0

t0 = tm.time()
reprobados = list(p.map(calcular_reprobados, chuncks))
t1 = tm.time()
tiempo_paralelo += t1 - t0
print('Tiempo calculo reprobados: ', t1-t0)

t0 = tm.time()
reprobados_totales = 0
for reprobado in reprobados:
    reprobados_totales += reprobado
t1 = tm.time()
tiempo_secuencial += t1-t0



t_final = tm.time()
print('\nTiempo de ejecucion secuencial: ', tiempo_secuencial)
print('Tiempo de ejecucion paralelo: ', tiempo_paralelo)
print('Tiempo de ejecucion total: ', t_final-t_inicial)

print('Cantidad de reprobados: ',reprobados_totales)
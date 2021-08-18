import time as tm

t0 = tm.time()

promedios_finales = []

cantidad_reprobados = 0

with open("notas.txt", 'r') as file:

    file.readline()

    for linea in file:

        notas = linea.strip().split(',')
        notas = [float(x) for x in notas]
        PC = (notas[0] + notas[1] + 2*notas[2] - min(notas[:3]))/3
        PT = (notas[3] + notas[4] + notas[5] + notas[6] + notas[7] + notas[8] + notas[9] - min(notas[3:]))/6
        PF = round(0.7*PC + 0.3*PT, 2)

        if notas[2] >= 4:
            PF = max(PF, 4)

        promedios_finales.append(PF)

        if PF < 4:
            cantidad_reprobados += 1

t1 = tm.time()


print('Tiempo de ejecucion: ', t1-t0)
print('Cantidad de reprobados: ', cantidad_reprobados)
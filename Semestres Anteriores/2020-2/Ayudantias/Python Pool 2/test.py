from multiprocessing import Pool
import time


def notas(l):
    nota = list(map(lambda x: float(x), l))
    NC = nota[0] + nota[1] + 2 * nota[2] - min(nota[0], nota[1], nota[2])
    NC = NC/3
    NT = sum(nota[3:]) - min(nota[3:])
    NT = NT/6
    NF = 0.7 * NC + 0.3 * NT
    if nota[2] >= 4:
        NF = max(4, NF)
    return NF


def reprueba(al):
    if al < 4:
        return False
    else:
        return True


alumnos = []
with open("notas.txt", "r", encoding="UTF-8") as arc:
    lineas = arc.readlines()
    for linea in lineas[1:]:
        alumnos.append(linea.strip().split(","))
t1 = time.time()
notass = list(map(lambda l: notas(l), alumnos))
print(notass[:10])
t2 = time.time()
print(t2-t1)

t1 = time.time()
p = Pool(2)
notas_ = p.map(notas, alumnos)
print("uwu")
t2 = time.time()
print(t2-t1)
p.close()

t1 = time.time()
p = Pool(2)
aprobados = p.filter(reprueba, notas_)
t2 = time.time()
print(t2-t1)
p.close()
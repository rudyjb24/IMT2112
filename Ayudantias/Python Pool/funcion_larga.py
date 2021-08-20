import time as tm
import math
import random as rd
from multiprocessing import Pool

def f(x):
    y = math.sqrt(x**2 + x*math.cos(x))
    z = (y*x + math.sin(x+2*y))**(1/3)
    output = x*y*z * math.tan(x+y+z)
    return output

lista = [rd.randint(0,100) for x in range(100000)]

t1 = tm.time()
[x for x in map(f, lista)]
t2 = tm.time()
print("\nTiempo sin workers: {} \n".format(round(t2-t1, 6)))

t1 = tm.time()
p = Pool(2)
p.map(f, lista)
t2 = tm.time()
print("Tiempo con 2 worker: {}\n".format(round(t2-t1, 6)))

t1 = tm.time()
p = Pool(4)
p.map(f, lista)
t2 = tm.time()
print("Tiempo con 4 worker: {}\n".format(round(t2-t1, 6)))

t1 = tm.time()
p = Pool(8)
p.map(f, lista)
t2 = tm.time()
print("Tiempo con 8 worker: {}\n".format(round(t2-t1, 6)))

t1 = tm.time()
p = Pool(12)
p.map(f, lista)
t2 = tm.time()
print("Tiempo con 12 worker: {}\n".format(round(t2-t1, 6)))

t1 = tm.time()
p = Pool(16)
p.map(f, lista)
t2 = tm.time()
print("Tiempo con 16 worker: {}\n".format(round(t2-t1, 6)))

t1 = tm.time()
p = Pool(24)
p.map(f, lista)
t2 = tm.time()
print("Tiempo con 24 worker: {}\n".format(round(t2-t1, 6)))

t1 = tm.time()
p = Pool(32)
p.map(f, lista)
t2 = tm.time()
print("Tiempo con 32 worker: {}\n".format(round(t2-t1, 6)))

t1 = tm.time()
p = Pool(64)
p.map(f, lista)
t2 = tm.time()
print("Tiempo con 64 worker: {}\n".format(round(t2-t1, 6)))
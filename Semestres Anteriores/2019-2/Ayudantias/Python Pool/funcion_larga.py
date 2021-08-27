import time as tm
import math
import random as rd
from multiprocessing import Pool

def f(x):
    y = x**2 + x*math.cos(x)
    z = (y*x + math.sin(x+2*y))**(1/3)
    output = x*y*z * math.tan(x+y+z)
    return output

p1 = Pool(1)
p2 = Pool(2)
p3 = Pool(3)
p4 = Pool(4)
p5 = Pool(8)
p6 = Pool(50)

lista = [rd.randint(0,100) for x in range(100000)]

t1 = tm.time()
[x for x in map(f, lista)]
t2 = tm.time()
print("\nTiempo sin workers: {} \n".format(round(t2-t1, 4)))

t1 = tm.time()
p1.map(f, lista)
t2 = tm.time()
print("Tiempo con 1 worker: {}\n".format(round(t2-t1, 4)))

t1 = tm.time()
p2.map(f, lista)
t2 = tm.time()
print("Tiempo con 2 worker: {}\n".format(round(t2-t1, 4)))

t1 = tm.time()
p3.map(f, lista)
t2 = tm.time()
print("Tiempo con 3 worker: {}\n".format(round(t2-t1, 4)))

t1 = tm.time()
p4.map(f, lista)
t2 = tm.time()
print("Tiempo con 4 worker: {}\n".format(round(t2-t1, 4)))

t1 = tm.time()
p5.map(f, lista)
t2 = tm.time()
print("Tiempo con 8 worker: {}\n".format(round(t2-t1, 4)))

t1 = tm.time()
p6.map(f, lista)
t2 = tm.time()
print("Tiempo con 50 worker: {}\n".format(round(t2-t1, 4)))
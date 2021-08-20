import random as rd
from multiprocessing import Pool

def f(x):
    if (x >= 4):
        return True
    else:
        return False


lista = [rd.randint(1,7) for x in range(10)]
print(lista)

p = Pool(2)
resultados = p.map(f, lista)
p.close()

print(sum(resultados))

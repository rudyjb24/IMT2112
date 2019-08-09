def f(x):
    return x+1

lista = [0,1,2,3,4,5,6,7,9]

print("\nMap 1")
map1 = map(f, lista)
print(map1)
print(type(map1))
print(list(map1))


print("\nMap 2")
map2 = map(lambda x: x+2, lista)
print(map2)
print(type(map2))
print(tuple(map2))



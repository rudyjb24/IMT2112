import multiprocessing

def f(x):
    print(multiprocessing.current_process()._identity[0])

p = multiprocessing.Pool(3)
p.map(f, range(6))
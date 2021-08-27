from multiprocessing import Pool
import pandas as pd
import numpy as np
import time as tm

def gen_pts(dim):
    return np.random.randn(dim)

def dist_centros(ex_centros, centros, pool):
    suma = 0
    for p1,p2 in zip(ex_centros, centros):
        suma += dist_pts_multi(p1,p2,pool)
    return suma

def label_multi(data,ctrs,pool):
    # ambos argumentos son arrays
    # [[d11,d12,..,d1k],[d21,d22,..,d2k],...,[dn1,dn2,..,dnk]]
    dists = np.array([[dist_pts_multi(data.loc[i], ctr, pool) for ctr in ctrs] for i in range(data.shape[0])])
    cercanos = pool.map(argmins, dists)
    return cercanos


def dist_pts_multi(p1,p2,pool):
    return np.sqrt(sum(pool.map(res, zip(p1,p2))))

def res(pts): 
    hol= (pts[0]-pts[1])**2
    return hol

def argmins(tup):
    return np.argmin(tup)

def prom_eje(datos):
    #viene separado por label
    return np.array(datos.mean())


def kmeans_multi(pool, n, dim, k, err =0.1):
    print("Hello")
    print(f'muestras \t= {n}\ndimension \t= {dim}\nn de centros \t= {k}\nmax. despl. \t= {err}')
    # generar puntos ; n,dim
    # pool = Pool(workers)
    data = pd.DataFrame(pool.map(gen_pts, (dim for _ in range(n))), columns=[f'x_{i}' for i in range(dim)])

    # # generar centros ; k, dim
    centros = np.array([ np.random.randn(dim) for i in range(k)])
    
    
    copia = data[[f'x_{i}' for i in range(dim)]]
    
    ex_centros = np.zeros(centros.shape)
    d = dist_centros(ex_centros, centros,pool)
    
    while d > err :
        # etiquetar puntos
        data['lab'] = label_multi(copia,centros,pool)
    
        # recalcular centros 
        ex_centros = centros.copy()

        centros = np.array(pool.map(prom_eje, [copia[data.lab == m] for m in range(k) ]))

        d = dist_centros(ex_centros, centros,pool)
    return centros


##############
if __name__ == "__main__":
    print("Esto puede tardar 2 minutos: n=1000, dim=2, k=3, err=0.01")
    t1=tm.time()
    wkrs = 4
    print("n de trabajadores: \t", wkrs)
    with Pool(wkrs) as pool:
        ctrs = kmeans_multi(pool,n=100, dim=2, k=3, err=0.01)
    t2=tm.time()
    print(ctrs)
    print(f'Obtenido en {t2-t1} segunds')
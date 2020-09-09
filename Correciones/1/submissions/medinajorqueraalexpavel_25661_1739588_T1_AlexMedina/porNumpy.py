import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time as tm


def label_numpy(data , centros):
    return np.array([((data-cen) ** 2).sum(axis = 1) for cen in centros]).argmin(axis = 0)

def dist_centros(centros, excentros):
    return np.sqrt(sum((np.array([((centros - excentros) ** 2).sum(axis = 1) for cen in centros]).min(axis = 0))**2))

def kmeans_np(n, dim, k, err =0.1):
    print(f'muestras \t= {n}\ndimension \t= {dim}\nn de centros \t= {k}\nmax. despl. \t= {err}')
    data = pd.DataFrame(np.random.randn(n, dim),columns=[f'x_{i}' for i in range(dim)])

    #
    # plt.scatter(data.x_0,data.x_1)
    # plt.title("iniciando puntos")
    # plt.show()
    #

    centros =  np.random.randn(k,dim)

    #
    # cols = [i for i in range(k)]
    # plt.scatter(data.x_0,data.x_1, alpha=0.3, marker="x")
    # plt.scatter(centros[:,0],centros[:,1], c=cols, s=100)
    # plt.title("iniciando centros")
    # plt.show()
    #
    
    copia = data[[f'x_{i}' for i in range(dim)]]
    
    ex_centros = np.zeros(centros.shape)
    d = dist_centros(ex_centros, centros)

    while d > err:

        labs = label_numpy(copia,centros)
        data['lab'] = labs
        
        #
        # plt.scatter(data.x_0, data.x_1, alpha = 0.3, marker = "x", c = labs)
        # plt.scatter(centros[:,0], centros[:,1], c = cols, s = 100)
        # plt.title("marcando puntos")
        # plt.show()
        #
        
        ex_centros = centros.copy()
        for m in range(k):
            centros[m] = np.array(data.groupby('lab').mean().loc[m])
            
        # #
        # plt.scatter(data.x_0,data.x_1, alpha=0.3, marker="x", c = labs)
        # plt.scatter(centros[:,0],centros[:,1], c=cols, s=100)
        # plt.title("recalculando centros")
        # plt.show()
        #
        
        d = dist_centros(ex_centros, centros)
        # print(d)
    return centros

######################
t1=tm.time()
ctrs = kmeans_np(n=1000,dim=2,k=4, err=0.01)
t2=tm.time()
print(ctrs)
print(f'Obtenido en {t2-t1} segunds')
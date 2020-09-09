import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time as tm


def dist(p1,p2):
    suma = 0
    for i in range(len(p1)):
        suma += (p1[i]-p2[i])**2
    return np.sqrt(suma)

def label_mala(data,ctrs):
    # ambos argumentos son arrays
    dists = np.array([dist(data,ctr) for ctr in ctrs])
    return dists.argmin()

def dist_centros(c1,c2):
    suma = 0
    for i in range(len(c1)):
        suma += dist(c1[i],c2[i])**2
    return suma

def kmeans_loop(n, dim, k, err =0.1):
    print(f'muestras \t= {n}\ndimension \t= {dim}\nn de centros \t= {k}\nmax. despl. \t= {err}')
    data = pd.DataFrame({f'x_{d}':[round(np.random.randn(),5) for _ in range(n)] for d in range(dim)} )
    
    ###v
    # data = pd.DataFrame({f'x_{d}':[round(np.random.randn(),5) for _ in range(n)] for d in range(dim)} )
    # plt.scatter(data.x_0,data.x_1)
    # plt.title("iniciando puntos")
    # plt.show()
    ##^
    
    centros = np.array([ np.random.randn(dim) for i in range(k)])
    
    ##v
    # cols = [i for i in range(k)]
    # plt.scatter(data.x_0,data.x_1, alpha=0.3, marker="x")
    # plt.scatter(centros[:,0],centros[:,1], c=cols, s=100)
    # plt.title("iniciando centros")
    # plt.show()
    ##^
    
    
    copia = data[[f'x_{i}' for i in range(dim)]]
    
    ex_centros = centros+12
    d = dist_centros(ex_centros, centros)

    while d > err:

        data['lab'] = [ label_mala(copia.loc[i],centros) for i in range(data.shape[0])] 
        
        ###v
        # plt.scatter(data.x_0,data.x_1, alpha=0.3, marker="x", c = labs)
        # plt.scatter(centros[:,0],centros[:,1], c=cols, s=100)
        # plt.title("marcando puntos")
        # plt.show()
        ##^
        
        ex_centros = centros.copy()
        
        for m in range(k):
            for i in range(dim):
                centros[m,i] = data[data.lab == m][f'x_{i}'].mean()
        ##v
        # plt.scatter(data.x_0,data.x_1, alpha=0.3, marker="x", c = labs)
        # plt.scatter(centros[:,0],centros[:,1], c=cols, s=100)
        # plt.title("recalculando centros")
        # plt.show()
        ###^
        
        d = dist_centros(ex_centros, centros)
        # print(d)
    return centros


###################
t1=tm.time()
ctrs = kmeans_loop(n=1000,dim=2,k=4, err=0.01)
t2=tm.time()
print(ctrs)
print(f'Obtenido en {t2-t1} segunds')
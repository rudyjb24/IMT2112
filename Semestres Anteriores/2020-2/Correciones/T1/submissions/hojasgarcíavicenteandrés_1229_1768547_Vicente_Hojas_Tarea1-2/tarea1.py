# Tarea 1 algoritmos paralelos en computacion cientifica
# Vicente Hojas


from multiprocessing import Pool
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import math
import time as tm

class ParallelFunctions:
    '''
    Incluye las funciones que se le aplican a cada chunck que se le pasa a map, 
    como Multiprocessing hace pickle de las funciones, de esta forma se pueden 
    mantener variables como C, m y k y se pueden pasar funciones que llamen a 
    otras funciones
    '''

    def __init__(self, C, m, k):
        self.m = m
        self.k = k
        self.C = C
        self.x = None
    
    def centre_of_mass(self, arr):
        '''devuelve la suma de los registros, para el centro de masa, la division 
        por el total de registros clasificados se hace en ParallelK'''
        X = arr[:, :-1]
        Y = arr[:, -1]
        centres = np.zeros((self.k, self.m))
        for i in range(self.k):
            # se filtran los registros
            masked = X[Y==i, :]
            if len(masked) != 0:
                centres[i, :] = np.sum(masked, axis=0)
        return centres

    def choosing_func(self, i):
        return np.linalg.norm(self.x-self.C[i, :])
    
    def classify(self, x):
        '''devuelve la clasificacion de un solo dato'''
        self.x = x
        y = min(range(self.k), key=self.choosing_func)
        return y


class ParallelK:

    def __init__(self, X, C, n_workers = 2, max_iter=1e6):
        self.X = X
        self.C = C
        self.max_iter = max_iter
        self.n, self.m = X.shape
        self.k, _ = C.shape
        self.iters = 0
        self.n_workers = n_workers
        self.pool = Pool(n_workers)
        self.functions = ParallelFunctions(self.C, self.m, self.k)

    def centres_of_mass(self, Y):
        '''devuelve los centros de masa de los datos self.X con clasificación 
        Y'''
        whole = np.concatenate((self.X, Y.reshape((self.n, 1))), axis=1)
        # dividimos el array completo por el nro de trabajadores
        size_chunks = int(self.n/self.n_workers)
        chunks = []
        for i in range(self.n_workers-1):
            chunks.append(whole[i*size_chunks:(i+1)*size_chunks, :])
        chunks.append(whole[(i+1)*size_chunks:, :])
        # calculamos los centros de masa de cada chunk de forma paralela
        chunks_centres = self.pool.map(
            self.functions.centre_of_mass, chunks
        )
        centres = sum(chunks_centres)
        # se divide la suma por el total de datos de cada clasificacion
        for i in range(self.k):
            total = np.count_nonzero(Y==i)
            if total != 0:
                centres[i] = centres[i]/total
        return centres

    def classify(self, Y_old=None):
        '''funcion recursiva que realiza la clasificacion hasta que esta se 
        mantenga constante al recalcular los centros C'''
        if self.iters >= self.max_iter:
            return Y_old
        if Y_old is not None:
            # se calculan los centros de masa de acuerdo a Y_old
            new_C = self.centres_of_mass(Y_old)
            # actualizamos los centros para clasificar
            self.C = new_C
            self.functions.C = new_C
        # clasificamos cada registro de forma paralela
        Y = np.array(self.pool.map(self.functions.classify, self.X))
        self.iters += 1
        if Y_old is not None: 
            if np.array_equal(Y, Y_old):
                # si la clasificacion con los nuevos centros es igual a la 
                # anterior, retorna la clasificacion anterior
                return Y
            else:
                # en caso contrario realiza otra iteracion
                return self.classify(Y)
        else:
            return self.classify(Y)

class ForLoopK:

    def __init__(self, X, C, m, n, k, max_iter=1e6):
        self.X = X.tolist()
        self.C = C.tolist()
        self.m = m
        self.n = n
        self.k = k
        self.iters = 0
        self.max_iter = max_iter
    
    def split_data(self, Y):
        '''divide los datos de acuerdo a la clasificacion Y'''
        split = [[] for i in range(self.k)]
        for i in range(self.n):
            split[Y[i]].append(self.X[i])
        return split

    def centres_of_mass(self, Y):
        '''calcula los centros  de masa de X de acuerdo a la clasificacion Y'''
        split = self.split_data(Y)
        centres = [[0 for i in range(m)] for j in range(k)]
        n_elements = [len(x) for x in split]
        for i in range(self.k):
            for x in split[i]:
                for j in range(self.m):
                    centres[i][j] += x[j]
            if n_elements[i] != 0:
                centres[i] = [x/n_elements[i] for x in centres[i]]
        return centres

    def classify(self, Y_old=None):
        '''clasifica los datos X de forma recursiva con k-means'''
        if self.iters >= self.max_iter:
            return Y_old
        if Y_old is not None:
            # se re-calculan los centros como los centros de masa de acuerdo a 
            # la clasificacion antigua
            new_C = self.centres_of_mass(Y_old)
            self.C = new_C
        Y = [0]*self.n
        for i in range(self.n):
            # se calcula la distancia euclidiana con loops
            d_min = 1e10
            # iteramos sobre los centros
            for j in range(self.k):
                squared_distance = 0
                # distancia euclidiana
                for s in range(self.m):
                    squared_distance += (self.X[i][s]-self.C[j][s])**2
                d = math.sqrt(squared_distance)
                if d < d_min:
                    # se escoge el centro con la menor distancia
                    d_min = d
                    Y[i] = j
        self.iters += 1
        if Y_old is not None: 
            if Y == Y_old:
                # si la clasificacion es igual a la antigua, se retorna
                return Y
            else:
                # en caso contrario se sigue clasificando
                return self.classify(Y)
        else:
            return self.classify(Y)

class NumPyK:

    def __init__(self, X, C, max_iter=1e6):
        self.X = X
        self.C = C
        self.n, self.m = X.shape
        self.k, _ = C.shape
        self.max_iter = max_iter
        self.iters = 0

    def centres_of_mass(self, Y):
        '''centros de masa de X para la clasificacion Y'''
        centres = np.zeros((self.k, self.m))
        for i in range(self.k):
            masked = self.X[Y==i, :]
            if len(masked) != 0:
                centres[i] = np.mean(masked, axis=0)
        return centres

    def classify(self, Y_old=None):
        '''clasificacion recursiva con k-means para X'''
        if self.iters >= self.max_iter:
            return Y_old
        if Y_old is not None:
            # se recalculan los centros como los centros de masa
            new_C = self.centres_of_mass(Y_old)
            self.C = new_C
        # se clasifican los datos X
        Y = np.array([
            min(
                range(self.k),
                key=lambda s: np.linalg.norm(self.C[s, :]-self.X[i, :])
            ) for i in range(self.n)
        ])
        self.iters += 1
        if Y_old is not None: 
            # si la clasificacion es igual a la antigua, se retorna
            if np.array_equal(Y, Y_old):
                return Y
            # en caso contrario se sigue clasificando
            else:
                return self.classify(Y)
        else:
            return self.classify(Y)

if __name__=='__main__':
    # tamaños de registros
    N = np.array([1000])
    T = np.zeros((len(N), 3))
    for (i, n) in enumerate(N):
        # numero de features
        m = 2
        # creacion del dataset
        X, Y = make_blobs(n_samples=n, n_features=m, centers=3)

        k = 3
        # centros
        C = 15*np.random.rand(k, m)

        # clasificadores
        loop_classifier = ForLoopK(X, C, m, n, k)
        numpy_classifier = NumPyK(X, C)
        pool_classifier = ParallelK(X, C, n_workers=2)
        # tiempo de clasificador con loops
        t1 = tm.time()
        Y1 = loop_classifier.classify()
        T[i, 0] = tm.time()-t1
        # tiempo de clasificador con numpy
        t2 = tm.time()
        Y2 = numpy_classifier.classify()
        T[i, 1] = tm.time()-t2
        # tiempo de clasificador con multiprocessing
        t3 = tm.time()
        Y3 = pool_classifier.classify()
        T[i, 2] = tm.time()-t3
        

    # graficamos los tiempos
    fig, ax = plt.subplots(1, 2, figsize=(10, 20))
    ax[0].set_title('Comparacion de tiempos')
    ax[0].plot(N, T[:, 0], label='for loops')
    ax[0].plot(N, T[:, 1], label='numpy')
    ax[0].plot(N, T[:, 2], label='multiprocessing')
    ax[0].set_xlabel('n registros')
    ax[0].set_ylabel('tiempo de ejecucion [s]')
    ax[0].legend()

    # scatter plot de los datos
    classified = loop_classifier.split_data(Y1)
    ax[1].set_title('scatterplot de datos clasificados')
    for i in range(len(classified)):
        data = np.array(classified[i])
        ax[1].scatter(data[:, 0], data[:, 1], label=f'grupo {i}')
    fig.show()
    plt.show(block=True)

    print('-'*20, '\n')
    print((
        'Nota final: en el plot se incluyen tiempos para m=2, pero en el informe'+
        ' se incluyen tiempos para m=20, donde Multiprocessing si es más'+
        ' rápido que numpy y numpy más rápido que los for loops, solo que'+
        ' preferí dejar m=2 como default porque es más rápido'
    ))

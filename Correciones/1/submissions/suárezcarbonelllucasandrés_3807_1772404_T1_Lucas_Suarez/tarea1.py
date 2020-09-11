from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import copy
import time
import math
from multiprocessing import Pool, cpu_count

n = 10000
dim = 20
centers = [[1, 1] + [0] * (dim-2), [-1, 1] + [0] * (dim-2),
           [-1, -1] + [0] * (dim-2), [1, -1] + [0] * (dim-2)]

data, y = make_blobs(n_samples=n, centers=centers, n_features=dim, random_state=3, cluster_std=0.1)
data_list = []
for x in data:
    vector_list = []
    for i in range(dim):
        vector_list.append(x[i])
    data_list.append(vector_list)

# Definimos función utilizada para categorizar cada punto en un cluster


def split_work(arg):
    temp_centers = [[0] * dim, [0] * dim, [0] * dim, [0] * dim]
    temp_classes = []
    temp_old_centers = arg[0]
    split_interval = arg[1]
    for k in range(split_interval[0], split_interval[1]):
        min_distance = 100000000000000000
        center_index = None
        for i in range(4):
            d = 0
            for j in range(dim):
                d += (temp_old_centers[i][j] - data_list[k][j])**2
            d = math.sqrt(d)
            if d < min_distance:
                min_distance = d
                center_index = i
        temp_classes.append(center_index)
        for i in range(dim):
            temp_centers[center_index][i] += data_list[k][i]
    return temp_centers, temp_classes


if __name__ == '__main__':
    print('Resultados con n = ', n)
    centers = [[1, 0] + [0] * (dim-2), [-1, 0] + [0] * (dim-2), [0, 1] + [0] * (dim-2), [0, -1] + [0] * (dim-2)]
    old_classes = [-1] * 1000
    stop = False
    initial_time = time.time()

    while not stop:
        # Primero determinamos los clusters de cada punto
        new_classes = []
        new_centers = [[0] * dim, [0] * dim, [0] * dim, [0] * dim]

        for vector in data_list:
            min_distance = 100000000000000000
            center_index = None
            for i in range(4):
                d = 0
                for j in range(dim):
                    d += (centers[i][j] - vector[j])**2
                d = math.sqrt(d)
                if d < min_distance:
                    min_distance = d
                    center_index = i
            new_classes.append(center_index)
            for i in range(dim):
                new_centers[center_index][i] += vector[i]
        
        if new_classes == old_classes:
            stop = True

        for i in range(4):
            class_count = new_classes.count(i)
            if class_count != 0:
                for h in range(dim):
                    new_centers[i][h] /= class_count

        old_classes = copy.copy(new_classes)
        centers = copy.copy(new_centers)
    end_time = time.time()
    python_loop_time = end_time - initial_time
    print(f'Tiempo total usando loops sin paralelización: {python_loop_time} s')
    plt.scatter(data[:, 0], data[:, 1], c=new_classes)
    plt.show()

# -------------------------------------------------------------------------------------------------------#
# Ahora usamos numpy para vectorizar...
    c_1 = np.array([1, 0] + [0] * (dim-2))
    c_2 = np.array([-1, 0] + [0.0] * (dim-2))
    c_3 = np.array([0, 1] + [0.0] * (dim-2))
    c_4 = np.array([0, -1] + [0.0] * (dim-2))
    centers = np.array([c_1, c_2, c_3, c_4])

    old_classes = [-1] * 1000
    stop = False
    initial_time = time.time()

    while not stop:
        # Primero determinamos los clusters de cada punto
        new_classes = []
        new_centers = np.array([np.zeros(dim), np.zeros(dim), np.zeros(dim), np.zeros(dim)])
        for vector in data:
            center_index = np.argmin(np.linalg.norm(centers - vector, axis=1))
            new_classes.append(center_index)
            new_centers[center_index] += vector
        
        for i in range(4):
            class_count = new_classes.count(i)
            if class_count != 0:
                new_centers[i] /= class_count

        if new_classes == old_classes:
            stop = True

        old_classes = copy.copy(new_classes)
        centers = copy.copy(new_centers)
    end_time = time.time()
    numpy_time = end_time - initial_time
    print(f'Tiempo total usando Numpy: {numpy_time} s')
    #plt.scatter(data[:, 0], data[:, 1], c=new_classes)
    #plt.show()



# -------------------------------------------------------------------------------------------------------#
# Por ultimo usamos multiprocessing...

    old_classes = [-1]
    centers = [[1, 0] + [0] * (dim-2), [-1, 0] + [0] * (dim-2), [0, 1] + [0] * (dim-2), [0, -1] + [0] * (dim-2)]
    stop = False
    # En mi computador son 4 nucleos
    nprocs = cpu_count()
    p = Pool(nprocs)
    # Separamos los datos en chunks
    intervals = []
    chunk_size = n / nprocs
    for i in range(nprocs):
        intervals.append([int(i * chunk_size), int((i+1)*chunk_size)])
    initial_time = time.time()
    stop = False
    while not stop:
        # Primero determinamos los clusters de cada punto
        to_proccess_data = [[centers, interval] for interval in intervals]
        centers_labels = p.map(split_work, to_proccess_data)
        new_classes = []
        new_centers = [[0] * dim, [0] * dim, [0] * dim, [0] * dim]
        for info in centers_labels:
            new_classes += info[1]
            for i in range(4):
                for l in range(dim):
                    new_centers[i][l] += info[0][i][l]
        for i in range(4):
            for l in range(dim):
                if new_classes.count(i) != 0:
                    new_centers[i][l] /= new_classes.count(i)

        if new_classes == old_classes:
            stop = True
        old_classes = copy.copy(new_classes)
        centers = copy.copy(new_centers)
    end_time = time.time()
    multi_time = end_time - initial_time
    print('Tiempo total usando multiprocessing: ', multi_time)
    plt.scatter(data[:, 0], data[:, 1], c=new_classes)
    #plt.savefig('plot1.png')
    plt.show()
# ---------------------------------------------------------------------------#

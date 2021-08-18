from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from method1 import LoopMeans, LoopMeans2
from method2 import NumPyMeans
from method3 import MultiMeans
import numpy as np
import random
import time


def plot_results(Xs, ys, num_res, colors, n):
    ax = [plt.subplot(int(f'23{i}')) for i in [1, 2, 3, 5, 6]]
    titles = ['Clusters Reales (Simulados)', 'Método 1: Sólo Loops',
              'Método 1: Loops y Map', 'Método 2', 'Método 3']

    for j in range(num_res + 1):

        pca = PCA(n_components=2)
        plot = pca.fit_transform(Xs[j])

        x1, x2 = np.array([x[0] for x in plot]), np.array([x[1] for x in plot])
        col = np.array([colors[ys[j][i] % len(colors)] for i in range(n)])
        lab = np.array([ys[j][i] for i in range(n)])

        ax[j].scatter(x1, x2, c=col, label=lab)
        ax[j].set_title(titles[j])

    plt.tight_layout()
    plt.show()

def method1(X_sim, centers_loop1, centers_loop2, dim, k, prt=True):
    X_list = X_sim.tolist()
    time_b = time.time()
    loopmeans = LoopMeans(X_list, centers_loop1, dim, k)
    result1_1 = loopmeans.solve()
    time_method1_1 = round(time.time() - time_b, 3)

    if prt:
        print('Tiempo Método 1 - Sólo Loops: ', time_method1_1)

    time_b = time.time()
    loopmeans2 = LoopMeans2(X_list, centers_loop2, dim, k)
    result1_2 = loopmeans2.solve()
    time_method1_2 = round(time.time() - time_b, 3)

    if prt:
        print('Tiempo Método 1 - Loops y Map: ', time_method1_2)

    return result1_1, result1_2, time_method1_1, time_method1_2

def method2(X_sim, centers_np, dim, k, prt=True):
    time_b = time.time()
    numpymeans = NumPyMeans(X_sim, centers_np, dim, k)
    result2 = numpymeans.solve()
    time_method2 = round(time.time() - time_b, 3)

    if prt:
        print('Tiempo Método 2: ', time_method2)

    return result2, time_method2

def method3(X_sim, centers_mp, dim, k, prt=True):
    X_list = X_sim.tolist()
    n_workers = 2
    time_b = time.time()
    multimeans = MultiMeans(X_list, centers_mp, dim, k, n_workers)
    result3 = multimeans.solve()
    time_method3 = round(time.time() - time_b, 3)

    if prt:
        print('Tiempo Método 3: ', time_method3)

    return result3, time_method3

def generate_centers(centers_loop, centers_numpy):
    centers_loop1 = centers_loop.copy()
    centers_loop2 = centers_loop.copy()
    centers_np = centers_numpy.copy()
    centers_mp = centers_loop.copy()
    return centers_loop1, centers_loop2, centers_np, centers_mp

def statistics(k_sim, dim, n, k):
    print('Generando estadísticas, esto toma harto tiempo.')

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ks = [3, 10, 100, 1000]
    ns = [1000, 5000, 10000, 20000]
    dims = [2, 10, 25, 100]

    times1_1, times1_2, times2, times3 = [], [], [], []

    print('Estadísticas variando k: Comenzando')

    # Primero, variamos el número de clusters.
    for ka in ks:
        print('k:', ka)
        X_sim, y_sim = make_blobs(n_samples=n, centers=ka,
                                  n_features=dim, random_state=27)

        c_list = random.sample([i for i in range(len(X_sim))], ka)
        centers_loop = [X_sim[c_list[i]].tolist() for i in range(len(c_list))]
        centers_numpy = [X_sim[c_list[i]] for i in range(len(c_list))]

        centers_loop1, centers_loop2, centers_np, centers_mp = generate_centers(
            centers_loop, centers_numpy)

        time1_1, time1_2 = method1(X_sim, centers_loop1,
                                centers_loop2, dim, ka, False)[2:]
        times1_1.append(time1_1)
        times1_2.append(time1_2)
        times2.append(method2(X_sim, centers_np, dim, ka, False)[1])
        times3.append(method3(X_sim, centers_mp, dim, ka, False)[1])

    print('Estadísticas variando k: Listo')

    ax[0].plot(ks, times1_1, label='Método 1: Sólo Loops')
    ax[0].plot(ks, times1_2, label='Método 1: Loops y Map')
    ax[0].plot(ks, times2, label='Método 2')
    ax[0].plot(ks, times3, label='Método 3')

    ax[0].set_title('Variación en número de clusters')
    ax[0].set_xlabel('k')
    ax[0].set_ylabel('Tiempo')
    times1_1, times1_2, times2, times3 = [], [], [], []

    print('Estadísticas variando dim: Comenzando')

    for dima in dims:
        print('dim:', dima)
        X_sim, y_sim = make_blobs(n_samples=n, centers=k_sim,
                          n_features=dima, random_state=27)

        c_list = random.sample([i for i in range(len(X_sim))], k)
        centers_loop = [X_sim[c_list[i]].tolist() for i in range(len(c_list))]
        centers_numpy = [X_sim[c_list[i]] for i in range(len(c_list))]

        centers_loop1, centers_loop2, centers_np, centers_mp = generate_centers(
            centers_loop, centers_numpy)

        time1_1, time1_2 = method1(X_sim, centers_loop1,
                                centers_loop2, dima, k, False)[2:]
        times1_1.append(time1_1)
        times1_2.append(time1_2)
        times2.append(method2(X_sim, centers_np, dima, k, False)[1])
        times3.append(method3(X_sim, centers_mp, dima, k, False)[1])

    print('Estadísticas variando dim: Listo')

    ax[1].plot(dims, times1_1, label='Método 1: Sólo Loops')
    ax[1].plot(dims, times1_2, label='Método 1: Loops y Map')
    ax[1].plot(dims, times2, label='Método 2')
    ax[1].plot(dims, times3, label='Método 3')

    ax[1].set_title('Variación en dimensión')
    ax[1].set_xlabel('dim')
    ax[1].set_ylabel('Tiempo')
    times1_1, times1_2, times2, times3 = [], [], [], []

    print('Estadísticas variando n: Comenzando')

    for na in ns:
        print('n:', na)
        X_sim, y_sim = make_blobs(n_samples=na, centers=k_sim,
                          n_features=dim, random_state=27)

        c_list = random.sample([i for i in range(len(X_sim))], k)
        centers_loop = [X_sim[c_list[i]].tolist() for i in range(len(c_list))]
        centers_numpy = [X_sim[c_list[i]] for i in range(len(c_list))]

        centers_loop1, centers_loop2, centers_np, centers_mp = generate_centers(
            centers_loop, centers_numpy)

        time1_1, time1_2 = method1(X_sim, centers_loop1,
                                centers_loop2, dim, k, False)[2:]
        times1_1.append(time1_1)
        times1_2.append(time1_2)
        times2.append(method2(X_sim, centers_np, dim, k, False)[1])
        times3.append(method3(X_sim, centers_mp, dim, k, False)[1])

    print('Estadísticas variando n: Listo')

    ax[2].plot(ns, times1_1, label='Método 1: Sólo Loops')
    ax[2].plot(ns, times1_2, label='Método 1: Loops y Map')
    ax[2].plot(ns, times2, label='Método 2')
    ax[2].plot(ns, times3, label='Método 3')

    ax[2].set_title('Variación en número de datos')
    ax[2].set_xlabel('n')
    ax[2].set_ylabel('Tiempo')

    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    plt.show()

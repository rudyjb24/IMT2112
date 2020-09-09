from sklearn.datasets import make_blobs
from matplotlib import colors as mc
import numpy as np
import functions
import random


# Parámetros.
random.seed(27)
dim = 100               # Dimensión de los datos (>= 2).
n = 1000                # Número de datos (>= 1000).
k = 100                 # Número de centros algoritmo (>= 3).
k_sim = 100             # Número de centros simulados.

X_sim, y_sim = make_blobs(n_samples=n, centers=k_sim,
                          n_features=dim, random_state=27)

# Elegimos k centros al azar, en base a los datos.
c_list = random.sample([i for i in range(len(X_sim))], k)
centers_loop = [X_sim[c_list[i]].tolist() for i in range(len(c_list))]
centers_numpy = [X_sim[c_list[i]] for i in range(len(c_list))]

centers_loop1, centers_loop2, centers_np, centers_mp = functions.generate_centers(
            centers_loop, centers_numpy)

# Método 1.
result1_1, result1_2 = functions.method1(X_sim, centers_loop1,
                                         centers_loop2, dim, k)[:2]

# Método 2.
result2 = functions.method2(X_sim, centers_np, dim, k)[0]

# Método 3.
result3 = functions.method3(X_sim, centers_mp, dim, k)[0]

# Resultados.
results = [result1_1, result1_2, result2, result3]

# Reducción a componentes significativas y
# generación de gráficos.
num_res = len(results)
Xs, ys = [X_sim], [y_sim]
colors = list(mc.CSS4_COLORS)
random.shuffle(colors)

for j in range(num_res):
    X, y = [], []
    for i in range(k):
        X += results[j][i]
        y += [i for j in range(len(results[j][i]))]
    Xs.append(np.array(X))
    ys.append(np.array(y))

functions.plot_results(Xs, ys, num_res, colors, n)

# Generación de estatísticas.
functions.statistics(k_sim, dim, n, k)

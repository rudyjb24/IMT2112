import numpy as np
import matplotlib.pyplot as plt


with open('residuos.txt', 'r') as file:
    res = np.array([float(x.strip()) for x in file.readlines()])

fig, ax = plt.subplots(figsize=(12, 5))
ax.set_xlabel('t')
ax.set_ylabel('$\|Ax - b\|^2$')
ax.set_yscale('log')
ax.plot(res)

plt.show()

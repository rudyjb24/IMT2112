import numpy as np
import matplotlib.pyplot as plt


with open('mu.txt', 'r') as file:
    mu = np.array([float(x.strip()) for x in file.readlines()])

error = np.array([abs(mu[i + 1] - mu[i]) for i in range(len(mu) - 1)])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.set_xlabel('k')
ax1.set_ylabel('$\mu_{k}$')
ax1.plot(mu)

ax2.set_xlabel('k')
ax2.set_ylabel('|$\mu_{k + 1} - \mu_{k}|$')
ax2.plot(error)

fig.tight_layout()
plt.show()
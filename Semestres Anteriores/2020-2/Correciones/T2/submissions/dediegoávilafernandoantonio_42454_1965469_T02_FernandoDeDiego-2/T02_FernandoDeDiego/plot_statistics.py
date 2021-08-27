import numpy as np
import matplotlib.pyplot as plt


with open('mu10.txt', 'r') as file:
    mu10 = np.array([float(x.strip()) for x in file.readlines()])

with open('mu100.txt', 'r') as file:
    mu100 = np.array([float(x.strip()) for x in file.readlines()])

with open('mu1000.txt', 'r') as file:
    mu1000 = np.array([float(x.strip()) for x in file.readlines()])

with open('times.txt', 'r') as file:
    times = np.array([float(x.strip()) for x in file.readlines()])

fig = plt.figure(figsize=(15, 10))

gs = fig.add_gridspec(2,3)
ax1 = fig.add_subplot(gs[0, 0], xlabel='k', ylabel='$\mu_{k}$',
    title='n = 10')
ax2 = fig.add_subplot(gs[0, 1], xlabel='k', ylabel='$\mu_{k}$',
    title='n = 100')
ax3 = fig.add_subplot(gs[0, 2], xlabel='k', ylabel='$\mu_{k}$',
    title='n = 1000')
ax4 = fig.add_subplot(gs[1, :], xlabel='threads', ylabel='time',
    title='Threads v/s Time (n = 100)')

ax1.plot(mu10)
ax2.plot(mu100)
ax3.plot(mu1000)
ax4.plot(times)

plt.tight_layout()
plt.show()
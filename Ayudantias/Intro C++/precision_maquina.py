import numpy as np


epsilon = np.float64(1)

while 1 + epsilon != 1:
    
    epsilon_copy = epsilon
    epsilon /= 2

print(epsilon_copy)
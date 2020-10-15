import numpy as np


class NumPyMeans:

    def __init__(self, data, centers, dim, k):
        self.data = data
        self.centers = centers
        self.dim = dim
        self.k = k
        
        self.labels = {}

    def dif_dicts(self, lab1, lab2):
        if len(lab1) == 0:
            return True
        for i in range(self.k):
            if len(lab1[i]) != len(lab2[i]):
                return True
            for j in range(len(lab1[i])):
                if not (lab1[i][j] == lab2[i][j]).all():
                    return True
        return False

    def new_centers(self):
        for i in range(self.k):
            if len(self.labels[i]) == 0:
                continue
            summ = np.zeros(self.dim)
            for point in self.labels[i]:
                summ += point
            summ /= len(self.labels[i])
            self.centers[i] = summ

    def change_labels(self):
        new = {i: [] for i in range(self.k)}
        for d in range(len(self.data)):

            near = 0
            dist = np.linalg.norm(self.centers[0] - self.data[d])
            for c in range(1, self.k):
                new_dist = np.linalg.norm(self.centers[c] - self.data[d])
                if new_dist < dist:
                    near = c
                    dist = new_dist
            new[near].append(self.data[d])

        boolean = self.dif_dicts(self.labels, new)
        self.labels = new
        return boolean

    def solve(self):
        cont = self.change_labels()
        while cont:
            self.new_centers()
            cont = self.change_labels()
        return self.labels

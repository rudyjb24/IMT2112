class LoopMeans:

    def __init__(self, data, centers, dim, k):
        self.data = data
        self.centers = centers
        self.dim = dim
        self.k = k

        self.labels = [[] for i in range(k)]

    def distance(self, x, y):
        summ = 0
        for i in range(self.dim):
            summ += pow(x[i] - y[i], 2)
        return summ

    def new_centers(self):
        for i in range(self.k):
            if len(self.labels[i]) == 0:
                continue
            summ = [0 for i in range(self.dim)]
            for point in self.labels[i]:
                for crd in range(len(point)):
                    summ[crd] += point[crd]
            for j in range(self.dim):
                summ[j] /= len(self.labels[i])
            self.centers[i] = summ

    def change_labels(self):
        new_labels = [[] for i in range(self.k)]
        for d in self.data:
            nearest = 0
            dist = self.distance(self.centers[0], d)
            for c in range(1, self.k):
                new_dist = self.distance(self.centers[c], d)
                if new_dist < dist:
                    nearest = c
                    dist = new_dist
            new_labels[nearest].append(d)
        boolean = self.labels != new_labels
        self.labels = new_labels
        return boolean

    def solve(self):
        cont = self.change_labels()
        while cont:
            self.new_centers()
            cont = self.change_labels()
        return self.labels


class LoopMeans2:

    def __init__(self, data, centers, dim, k):
        self.data = data
        self.centers = centers
        self.dim = dim
        self.k = k

        self.labels = [[] for i in range(k)]
    
    def squared(self, X):
        y = X[0] - X[1]
        return y * y
    
    def sum_vector(self, v):
        return v[0] + v[1]
    
    def new_centers(self, i):
        if len(self.labels[i]) == 0:
            return self.centers[i]
        vector = [0 for i in range(self.dim)]
        for point in self.labels[i]:
            X = list(zip(point, vector))
            vector = list(map(self.sum_vector, X))
        length = len(self.labels[i])
        vector = [v / length for v in vector]
        return vector

    def lower_distance(self, X):
        X = zip(X[0], X[1])
        return sum(map(self.squared, X))

    def closests(self, d):
        X = zip(self.centers, [d for i in range(self.k)])
        distances = list(map(self.lower_distance, X))
        nearest = distances.index(min(distances))
        return nearest, d

    def change_labels(self):
        self.new_labels = [[] for i in range(self.k)]
        tuples = list(map(self.closests, self.data))
        for t in tuples:
            self.new_labels[t[0]].append(t[1])
        boolean = self.labels != self.new_labels
        self.labels = self.new_labels
        return boolean

    def solve(self):
        cont = self.change_labels()
        while cont:
            self.centers = list(map(self.new_centers, [i for i in range(self.k)]))
            cont = self.change_labels()
        return self.labels

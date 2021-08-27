import multiprocessing


class MultiMeans:

    def __init__(self, data, centers, dim, k, n=2):
        self.data = data
        self.centers = centers
        self.dim = dim
        self.k = k
        self.n = n

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

    def closest(self, d):
        X = zip(self.centers, [d for i in range(self.k)])
        distances = list(map(self.lower_distance, X))
        nearest = distances.index(min(distances))
        return nearest, d

    def my_append(self, t):
        # Altamente ineficiente (sin este demora menos).
        self.new_labels[t[0]].append(t[1])

    def change_labels(self, p):
        self.new_labels = [[]for i in range(self.k)]
        tuples = list(p.map(self.closest, self.data))

        # Descomentar si se quiere código 100% paralelizado.
        # Sin embargo, hacer append paralelizado no conviene.

        # manager = multiprocessing.Manager()
        # self.new_labels = [manager.list() for i in range(self.k)]
        # p.map(self.my_append, tuples)
        # self.new_labels = [list(l) for l in self.new_labels]
        
        # Comentar próximas dos líneas si se descomentan las
        # anteriores.
        for t in tuples:
            self.new_labels[t[0]].append(t[1])
        boolean = self.labels != self.new_labels
        self.labels = self.new_labels
        return boolean

    def solve(self):
        p = multiprocessing.Pool(self.n)
        cont = self.change_labels(p)
        while cont:
            self.centers = list(p.map(self.new_centers, [i for i in range(self.k)]))
            cont = self.change_labels(p)
        return self.labels

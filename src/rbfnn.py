import numpy as np
from kmeans import kmeans
from adaline import Adaline


class RBFNN:
    def __init__(self, k, eta=0.1, epocas=2000, semilla=42):
        self.k = k
        self.eta = eta
        self.epocas = epocas
        self.semilla = semilla

    def _calcular_sigma(self):
        d_max = 0.0
        for i in range(self.k):
            for j in range(self.k):
                d = np.sqrt(np.sum((self.centroides[i] - self.centroides[j]) ** 2))
                if d > d_max:
                    d_max = d
        if d_max == 0:
            return 1.0
        return d_max / np.sqrt(2 * self.k)

    def _construir_Z(self, X):
        Z = np.zeros((X.shape[0], self.k))
        for j in range(self.k):
            dist_sq = np.sum((X - self.centroides[j]) ** 2, axis=1)
            Z[:, j] = np.exp(-dist_sq / (2 * self.sigma ** 2))
        return Z

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.centroides, _ = kmeans(X, self.k, semilla=self.semilla)
        self.sigma = self._calcular_sigma()
        Z = self._construir_Z(X)
        self.adaline = Adaline(eta=self.eta, epocas=self.epocas, semilla=self.semilla).fit(Z, y)
        self.coste_ = self.adaline.coste_
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        Z = self._construir_Z(X)
        return self.adaline.predict(Z)

import numpy as np


class Adaline:
    def __init__(self, eta=0.1, epocas=2000, semilla=42):
        self.eta = eta
        self.epocas = epocas
        self.semilla = semilla

    def _net_input(self, X):
        return X @ self.w_ + self.b_

    def _activation(self, net_input):
        return net_input

    def _quantization(self, activation):
        return activation

    def fit(self, X, y):
        rng = np.random.default_rng(self.semilla)
        self.w_ = rng.normal(0, 0.01, X.shape[1])
        self.b_ = 0.0
        n = X.shape[0]
        self.coste_ = []
        for _ in range(self.epocas):
            salida = self._activation(self._net_input(X))
            error = y - salida
            self.w_ += self.eta * (X.T @ error) / n
            self.b_ += self.eta * np.mean(error)
            self.coste_.append(0.5 * np.mean(error ** 2))
        return self

    def predict(self, X):
        return self._quantization(self._activation(self._net_input(X)))

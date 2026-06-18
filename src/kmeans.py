import numpy as np


def distancia_euclidea(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def asignar_clusters(X, centroides):
    etiquetas = np.zeros(len(X), dtype=int)
    for i, punto in enumerate(X):
        distancias = [distancia_euclidea(punto, c) for c in centroides]
        etiquetas[i] = int(np.argmin(distancias))
    return etiquetas


def recalcular_centroides(X, etiquetas, k, rng):
    nuevos = []
    for j in range(k):
        puntos = X[etiquetas == j]
        if len(puntos) == 0:
            nuevos.append(X[rng.integers(0, len(X))])
        else:
            nuevos.append(puntos.mean(axis=0))
    return np.array(nuevos)


def kmeans(X, k, max_iter=100, semilla=42):
    rng = np.random.default_rng(semilla)
    indices = rng.choice(len(X), k, replace=False)
    centroides = X[indices].astype(np.float64)
    etiquetas = np.zeros(len(X), dtype=int)
    for _ in range(max_iter):
        etiquetas = asignar_clusters(X, centroides)
        nuevos = recalcular_centroides(X, etiquetas, k, rng)
        if np.allclose(centroides, nuevos):
            break
        centroides = nuevos
    return centroides, etiquetas

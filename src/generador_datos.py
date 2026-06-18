import numpy as np
import pandas as pd


def funcion_objetivo(x, y):
    return np.sin(np.sqrt(x * x + y * y))


def generar_muestras(n_muestras=1600, limite=5.0, semilla=42):
    rng = np.random.default_rng(semilla)
    x = rng.uniform(-limite, limite, n_muestras)
    y = rng.uniform(-limite, limite, n_muestras)
    etiqueta = funcion_objetivo(x, y)
    return pd.DataFrame({"x": x, "y": y, "etiqueta": etiqueta})

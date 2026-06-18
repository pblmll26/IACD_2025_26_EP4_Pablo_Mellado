import numpy as np
from generador_datos import generar_muestras
from rbfnn import RBFNN
from plot3d import plot_3d


def mse(y_real, y_pred):
    return np.mean((y_real - y_pred) ** 2)


def main():
    datos = generar_muestras(1600)
    X = datos[["x", "y"]].values
    y = datos["etiqueta"].values

    print("Datos generados correctamente")
    print(datos.head())

    print("\nEntrenando RBFNN...")
    modelo = RBFNN(k=30, eta=0.1, epocas=3000).fit(X, y)
    print("Entrenamiento finalizado")
    print("sigma =", modelo.sigma)

    pred = modelo.predict(X)
    print("\nMSE:", mse(y, pred))

    plot_3d(modelo)


if __name__ == "__main__":
    main()

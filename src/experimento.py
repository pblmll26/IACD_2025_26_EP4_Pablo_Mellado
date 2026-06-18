import os
import numpy as np
import matplotlib.pyplot as plt
from generador_datos import generar_muestras
from rbfnn import RBFNN


def mse(y_real, y_pred):
    return np.mean((y_real - y_pred) ** 2)


def dividir_train_test(X, y, proporcion=0.8, semilla=42):
    rng = np.random.default_rng(semilla)
    idx = rng.permutation(len(X))
    corte = int(proporcion * len(X))
    return X[idx[:corte]], X[idx[corte:]], y[idx[:corte]], y[idx[corte:]]


RESULTADOS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "resultados")


def ejecutar_experimento():
    datos = generar_muestras(1600)
    X = datos[["x", "y"]].values
    y = datos["etiqueta"].values
    X_train, X_test, y_train, y_test = dividir_train_test(X, y)

    lista_k = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100]
    errores_train, errores_test = [], []
    historiales = {}

    for k in lista_k:
        modelo = RBFNN(k=k, eta=0.1, epocas=2000).fit(X_train, y_train)
        e_train = mse(y_train, modelo.predict(X_train))
        e_test = mse(y_test, modelo.predict(X_test))
        errores_train.append(e_train)
        errores_test.append(e_test)
        historiales[k] = modelo.coste_
        print("K =", k, " MSE train =", round(e_train, 6), " MSE test =", round(e_test, 6))

    mejor_k = lista_k[int(np.argmin(errores_test))]
    print("\nMejor K =", mejor_k)

    os.makedirs(RESULTADOS, exist_ok=True)

    plt.figure(figsize=(9, 5))
    plt.plot(lista_k, errores_train, "o-", label="entrenamiento")
    plt.plot(lista_k, errores_test, "s-", label="test")
    plt.xlabel("Numero de neuronas ocultas (K)")
    plt.ylabel("MSE")
    plt.yscale("log")
    plt.title("RBFNN: numero de neuronas vs MSE")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTADOS, "MSE_vs_neuronas.png"))

    plt.figure(figsize=(9, 5))
    for k in [lista_k[0], lista_k[len(lista_k) // 2], mejor_k]:
        plt.plot(historiales[k], label=f"K = {k}")
    plt.xlabel("Epoca")
    plt.ylabel("J(w) = 1/2 * mean((y - phi(z))^2)")
    plt.yscale("log")
    plt.title("Evolucion del coste J(w) (regla Widrow-Hoff)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTADOS, "coste_Jw.png"))

    plt.show()


if __name__ == "__main__":
    ejecutar_experimento()

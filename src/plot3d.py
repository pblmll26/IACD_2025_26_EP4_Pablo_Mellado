import os
import numpy as np
import matplotlib.pyplot as plt
from generador_datos import funcion_objetivo


RESULTADOS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "resultados")


def plot_3d(modelo, limite=5.0, resolucion=60):
    xx = np.linspace(-limite, limite, resolucion)
    yy = np.linspace(-limite, limite, resolucion)
    XX, YY = np.meshgrid(xx, yy)
    rejilla = np.column_stack([XX.ravel(), YY.ravel()])
    Z_real = funcion_objetivo(XX, YY)
    Z_pred = modelo.predict(rejilla).reshape(XX.shape)
    Z_err = Z_real - Z_pred

    fig = plt.figure(figsize=(14, 4.5))
    ax1 = fig.add_subplot(131, projection="3d")
    ax1.plot_surface(XX, YY, Z_real, cmap="coolwarm")
    ax1.set_title("Funcion real")
    ax2 = fig.add_subplot(132, projection="3d")
    ax2.plot_surface(XX, YY, Z_pred, cmap="coolwarm")
    ax2.set_title(f"Aproximacion RBF (K = {modelo.k})")
    ax3 = fig.add_subplot(133, projection="3d")
    ax3.plot_surface(XX, YY, Z_err, cmap="seismic")
    ax3.set_title("Error (real - predicho)")
    plt.tight_layout()
    os.makedirs(RESULTADOS, exist_ok=True)
    plt.savefig(os.path.join(RESULTADOS, "superficies_3d.png"))
    plt.show()

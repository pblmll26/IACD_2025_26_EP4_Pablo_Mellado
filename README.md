# Red Neuronal RBF para aproximar f(x,y) = sin(√(x²+y²))

**Trabajo Práctico — Módulo 4 — Inteligencia Artificial para Ciencia de Datos (ULPGC)**

## 1. Objetivo

Crear una red neuronal RBF que aprenda a imitar la función `f(x,y) = sin(√(x²+y²))` en el cuadrado de
-5 a 5. La función dibuja unas ondas en forma de círculos, y queremos que la red las reproduzca.

## 2. Cómo funciona la red

La red tiene 3 pasos:

1. **K-means (centros):** repartimos K puntos "centro" por la zona donde están los datos. Cada centro
   será el corazón de una "campana" (función gaussiana).
2. **Funciones gaussianas:** cada neurona mide lo cerca que está un punto de su centro. Cerca → valor
   alto; lejos → valor casi cero.
3. **ADALINE (salida):** combina esas campanas con unos pesos para dar el resultado final. Los pesos se
   ajustan poco a poco bajando el error (regla de Widrow-Hoff).

En resumen: **datos → k-means → gaussianas → ADALINE → resultado.**

## 3. Datos

Generamos **1600 puntos** al azar dentro del cuadrado y calculamos su valor real con la función. Usamos
el **80% para entrenar** y el **20% para comprobar** que la red funciona también con datos nuevos.

## 4. El experimento

Probamos distintos números de neuronas (K) y medimos el error (MSE). La pregunta es: **¿cuántas
neuronas necesita la red para acertar?**

| Neuronas (K) | Error entrenamiento | Error test |
|---:|---:|---:|
| 5   | 0,229 | 0,217 |
| 10  | 0,110 | 0,106 |
| 20  | 0,084 | 0,083 |
| 30  | 0,038 | 0,036 |
| 50  | 0,008 | 0,008 |
| 75  | 0,005 | 0,005 |
| 100 | 0,005 | 0,004 |

## 5. Qué vemos en los resultados

- **Con pocas neuronas la red falla mucho** (con 5 el error es alto): no le dan las campanas para
  dibujar todas las ondas.
- **Cuantas más neuronas, menos error.** El error baja rápido.
- **A partir de unas 50 neuronas casi no mejora.** De 50 a 100 el error apenas cambia. Añadir más
  neuronas ya no compensa.
- **No hay sobreajuste:** el error con los datos de prueba es casi igual que con los de entrenamiento,
  así que la red generaliza bien.

## 6. Conclusión

La red RBF aproxima bien la función. El número de neuronas y el error van de la mano: más neuronas =
menos error, pero solo hasta un punto. **Un buen valor es K ≈ 50**, porque da muy poco error sin gastar
más neuronas de la cuenta.

## 7. Estructura del proyecto

```
IACD-EP4/
├── README.md
├── src/
│   ├── generador_datos.py   # genera los datos y la función f(x, y)
│   ├── kmeans.py            # algoritmo k-means (fase no supervisada)
│   ├── adaline.py           # ADALINE Widrow-Hoff (fase supervisada)
│   ├── rbfnn.py             # red RBF: k-means + gaussianas + ADALINE
│   ├── experimento.py       # estudio del nº de neuronas K frente al error
│   ├── plot3d.py            # gráficas 3D (real, aproximación y error)
│   └── main.py              # entrena un modelo y muestra las superficies
└── resultados/              # gráficas generadas
```

## 8. Cómo ejecutarlo

Necesitas Python con **NumPy, pandas y Matplotlib**. Desde la carpeta `src/`:

```bash
cd src
python main.py          # entrena un modelo y muestra/guarda las superficies 3D
python experimento.py   # estudio K vs error y evolución del entrenamiento
```

## 9. Gráficas generadas

- `resultados/MSE_vs_neuronas.png`: el error según el número de neuronas.
- `resultados/coste_Jw.png`: cómo baja el error mientras entrena.
- `resultados/superficies_3d.png`: la función real, la que hace la red y la diferencia entre ambas.

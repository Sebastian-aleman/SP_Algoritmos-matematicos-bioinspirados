"""Matriz de costos (distancia euclidiana sin redondear) y su exportación."""

import csv
import math


def calcular_matriz(coordenadas):
    """Devuelve la matriz c[i][j] de distancias euclidianas sin redondear."""
    n = len(coordenadas)
    matriz = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi = coordenadas[i]
            xj, yj = coordenadas[j]
            matriz[i][j] = matriz[j][i] = math.hypot(xi - xj, yi - yj)
    return matriz


def guardar_matriz_csv(matriz, ruta):
    """Guarda la matriz con precisión completa y etiquetas 0..n-1."""
    n = len(matriz)
    with open(ruta, "w", newline="", encoding="utf-8") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow([""] + list(range(n)))
        for i in range(n):
            escritor.writerow([i] + [repr(c) for c in matriz[i]])


def guardar_matriz_tex(matriz, ruta):
    """Guarda la matriz como tabla booktabs con 2 decimales."""
    n = len(matriz)
    lineas = [
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{c" + "r" * n + "}",
        r"\toprule",
        "$c_{ij}$ & " + " & ".join(str(j) for j in range(n)) + r" \\",
        r"\midrule",
    ]
    for i in range(n):
        lineas.append(f"{i} & " + " & ".join(f"{c:.2f}" for c in matriz[i]) + r" \\")
    lineas += [r"\bottomrule", r"\end{tabular}%", "}"]
    with open(ruta, "w", encoding="utf-8") as archivo:
        archivo.write("\n".join(lineas) + "\n")

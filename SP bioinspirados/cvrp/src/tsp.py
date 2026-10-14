"""TSP exacto (Held-Karp) y partición óptima (split) de una gran ruta."""

import math


def held_karp(matriz):
    """TSP exacto sobre todos los nodos. Devuelve (longitud, clientes en orden)."""
    n = len(matriz)
    N = n - 1
    dp = {(1 << (j - 1), j): (matriz[0][j], None) for j in range(1, n)}
    for mascara in range(1, 1 << N):
        for j in range(1, n):
            if (mascara, j) not in dp:
                continue
            costo = dp[(mascara, j)][0]
            for k in range(1, n):
                if mascara >> (k - 1) & 1:
                    continue
                nuevo = (mascara | 1 << (k - 1), k)
                valor = costo + matriz[j][k]
                if nuevo not in dp or valor < dp[nuevo][0]:
                    dp[nuevo] = (valor, (mascara, j))
    llena = (1 << N) - 1
    z, ultimo = min((dp[(llena, j)][0] + matriz[j][0], j) for j in range(1, n))
    recorrido, estado = [], (llena, ultimo)
    while estado is not None:
        recorrido.append(estado[1])
        estado = dp[estado][1]
    return z, recorrido[::-1]


def split(gran_ruta, matriz, demandas, capacidad):
    """Partición óptima de una gran ruta en rutas factibles (Prins, 2004)."""
    m = len(gran_ruta)
    V = [math.inf] * (m + 1)
    P = [0] * (m + 1)
    V[0] = 0.0
    for i in range(m):
        carga = 0
        for j in range(i, m):
            carga += demandas[gran_ruta[j]]
            if carga > capacidad:
                break
            d = (matriz[0][gran_ruta[i]]
                 + sum(matriz[gran_ruta[k]][gran_ruta[k + 1]] for k in range(i, j))
                 + matriz[gran_ruta[j]][0])
            if V[i] + d < V[j + 1]:
                V[j + 1] = V[i] + d
                P[j + 1] = i
    rutas, j = [], m
    while j > 0:
        rutas.append(gran_ruta[P[j]:j])
        j = P[j]
    return V[m], rutas[::-1]
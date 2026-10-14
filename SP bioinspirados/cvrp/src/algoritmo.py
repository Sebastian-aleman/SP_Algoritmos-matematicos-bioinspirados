"""Algoritmo propuesto: Clarke-Wright + búsqueda local."""

import statistics
import time

from src.busqueda_local import busqueda_local
from src.clarke_wright import clarke_wright
from src.matriz_costos import calcular_matriz


def resolver_clarke_wright(instancia):
    """Clarke-Wright solo, partiendo de la instancia. Devuelve las rutas."""
    matriz = calcular_matriz(instancia["coordenadas"])
    rutas, _ = clarke_wright(matriz, instancia["demandas"], instancia["capacidad"])
    return rutas


def resolver_cvrp(instancia):
    """Clarke-Wright + búsqueda local. Devuelve (rutas, movimientos)."""
    matriz = calcular_matriz(instancia["coordenadas"])
    rutas, _ = clarke_wright(matriz, instancia["demandas"], instancia["capacidad"])
    return busqueda_local(rutas, matriz, instancia["demandas"], instancia["capacidad"])


def medir_tiempo(funcion, instancia, repeticiones=100):
    """Mediana del tiempo (s) de `repeticiones` ejecuciones de funcion(instancia)."""
    tiempos = []
    for _ in range(repeticiones):
        inicio = time.perf_counter()
        funcion(instancia)
        tiempos.append(time.perf_counter() - inicio)
    return statistics.median(tiempos)

def medir_tiempo_exacto(matriz, demandas, capacidad, repeticiones=5):
    """Mediana del tiempo de solve del modelo exacto en `repeticiones` corridas."""
    from src.modelo_exacto import resolver_modelo_exacto
    tiempos = [resolver_modelo_exacto(matriz, demandas, capacidad)[1]["tiempo"]
               for _ in range(repeticiones)]
    return statistics.median(tiempos)

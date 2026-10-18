"""Algoritmo genético para el CVRP con cromosoma de gran ruta y split óptimo (Prins, 2004).

El cromosoma es una permutación de los clientes 1..n sin depósitos; su aptitud
es el costo de la partición óptima (split) en rutas factibles. Con
prob_busqueda_local > 0 se obtiene la versión memética: los hijos se mejoran
con busqueda_local y se re-codifican concatenando sus rutas.
"""

import math
import random
import sys
import time

from src.busqueda_local import busqueda_local

# Valores calibrados en main_calibracion.py; un valor cambia solo si reduce las
# evaluaciones hasta el mejor en más de 2 errores estándar. generaciones y paciencia
# se recortaron a la convergencia observada (se confirma en main_entrega2.py).
# tam_poblacion regresa a 50 porque, combinada con torneo 5, elitismo 5 y p_m = 0.05,
# la población de 20 convergió de forma prematura (90 % de éxito en la confirmación).
PARAMETROS_GA = {
    "tam_poblacion": 50,
    "generaciones": 100,
    "prob_cruza": 0.9,
    "prob_mutacion": 0.05,
    "tam_torneo": 5,
    "elitismo": 5,
    "paciencia": 50,
    "prob_busqueda_local": 0.0,
}

TOLERANCIA_MEJORA = 1e-9
INTENTOS_POR_INDIVIDUO = 50  # evita ciclos infinitos si no hay suficientes permutaciones distintas
SUMA_COMPENSADA = sys.version_info >= (3, 12)  # sum() de floats usa la suma de Neumaier


def split_rapido(gran_ruta, matriz, demandas, capacidad):
    """Partición óptima de una gran ruta en O(n^2) (Prins, 2004).

    Igual que tsp.split, pero la distancia del tramo gran_ruta[i..j] se acumula
    de forma incremental en vez de recalcularse con sum(). Para que el costo y
    las rutas sean bit a bit los mismos, la acumulación reproduce la suma
    compensada de Neumaier que usa sum() desde Python 3.12.
    """
    m = len(gran_ruta)
    V = [math.inf] * (m + 1)
    P = [0] * (m + 1)
    V[0] = 0.0
    for i in range(m):
        primero = gran_ruta[i]
        carga = 0
        suma = compensacion = 0.0  # distancia de gran_ruta[i] a gran_ruta[j] sin el depósito
        for j in range(i, m):
            cliente = gran_ruta[j]
            carga += demandas[cliente]
            if carga > capacidad:
                break
            if j > i:
                arco = matriz[gran_ruta[j - 1]][cliente]
                total = suma + arco
                if SUMA_COMPENSADA:
                    if abs(suma) >= abs(arco):
                        compensacion += (suma - total) + arco
                    else:
                        compensacion += (arco - total) + suma
                suma = total
            tramo = suma + compensacion if compensacion and math.isfinite(compensacion) else suma
            d = matriz[0][primero] + tramo + matriz[cliente][0]
            if V[i] + d < V[j + 1]:
                V[j + 1] = V[i] + d
                P[j + 1] = i
    rutas, j = [], m
    while j > 0:
        rutas.append(gran_ruta[P[j]:j])
        j = P[j]
    return V[m], rutas[::-1]


def seleccion_torneo(poblacion, tam_torneo, rng):
    """Devuelve el cromosoma de menor costo entre tam_torneo individuos al azar."""
    participantes = rng.sample(poblacion, min(tam_torneo, len(poblacion)))
    return min(participantes, key=lambda individuo: individuo[0])[1]


def cruza_ox(padre_1, padre_2, rng):
    """Order crossover: copia un tramo de padre_1 y completa en el orden de padre_2.

    El relleno empieza después del segundo punto de corte y da la vuelta al final.
    """
    n = len(padre_1)
    if n < 2:
        return list(padre_1)
    i, j = sorted(rng.sample(range(n), 2))
    hijo = [None] * n
    hijo[i:j + 1] = padre_1[i:j + 1]
    copiados = set(padre_1[i:j + 1])
    orden_padre_2 = [padre_2[(j + 1 + k) % n] for k in range(n)]
    restantes = [cliente for cliente in orden_padre_2 if cliente not in copiados]
    for k, cliente in enumerate(restantes):
        hijo[(j + 1 + k) % n] = cliente
    return hijo


def mutar(cromosoma, rng):
    """Aplica al azar intercambio, inversión de tramo o inserción."""
    hijo = list(cromosoma)
    if len(hijo) < 2:
        return hijo
    a, b = rng.sample(range(len(hijo)), 2)
    operador = rng.choice(("intercambio", "inversion", "insercion"))
    if operador == "intercambio":
        hijo[a], hijo[b] = hijo[b], hijo[a]
    elif operador == "inversion":
        i, j = min(a, b), max(a, b)
        hijo[i:j + 1] = reversed(hijo[i:j + 1])
    else:
        hijo.insert(b, hijo.pop(a))
    return hijo


def algoritmo_genetico(matriz, demandas, capacidad, semilla=None, **parametros):
    """Algoritmo genético generacional con elitismo para el CVRP.

    Los parámetros no indicados toman su valor de PARAMETROS_GA. Devuelve un
    diccionario con: rutas, costo, gran_ruta, historial_mejor e
    historial_promedio (posición 0 = población inicial, posición g = tras la
    generación g), generaciones (ejecutadas), generacion_mejor, evaluaciones
    (llamadas a split_rapido), evaluaciones_hasta_mejor y tiempo_hasta_mejor
    (al evaluar el cromosoma de la mejor solución final), tiempo (s) y parametros.
    """
    desconocidos = set(parametros) - set(PARAMETROS_GA)
    if desconocidos:
        raise TypeError(f"Parámetros desconocidos: {', '.join(sorted(desconocidos))}")
    p = {**PARAMETROS_GA, **parametros}
    tam_poblacion = p["tam_poblacion"]
    if not 0 <= p["elitismo"] <= tam_poblacion:
        raise ValueError("elitismo debe estar entre 0 y tam_poblacion")

    inicio = time.perf_counter()
    rng = random.Random(semilla)
    clientes = list(range(1, len(matriz)))
    evaluaciones = 0

    def evaluar(cromosoma):
        """Devuelve (costo, cromosoma); con búsqueda local el cromosoma se re-codifica."""
        nonlocal evaluaciones
        costo, rutas = split_rapido(cromosoma, matriz, demandas, capacidad)
        evaluaciones += 1
        if p["prob_busqueda_local"] > 0 and rng.random() < p["prob_busqueda_local"]:
            rutas, _ = busqueda_local(rutas, matriz, demandas, capacidad)
            cromosoma = [cliente for ruta in rutas for cliente in ruta]
            costo, _ = split_rapido(cromosoma, matriz, demandas, capacidad)
            evaluaciones += 1
        return costo, cromosoma

    def agregar(poblacion, vistos, cromosoma):
        """Evalúa y agrega el cromosoma si no es un clon. Devuelve si se agregó."""
        if tuple(cromosoma) in vistos:
            return False
        costo, cromosoma = evaluar(cromosoma)
        if tuple(cromosoma) in vistos:
            return False
        vistos.add(tuple(cromosoma))
        # Individuo: (costo, cromosoma, evaluaciones y segundos al momento de crearlo)
        poblacion.append((costo, cromosoma, evaluaciones, time.perf_counter() - inicio))
        return True

    # Población inicial aleatoria sin repetidos
    poblacion, vistos = [], set()
    for _ in range(tam_poblacion * INTENTOS_POR_INDIVIDUO):
        if len(poblacion) == tam_poblacion:
            break
        cromosoma = clientes[:]
        rng.shuffle(cromosoma)
        agregar(poblacion, vistos, cromosoma)
    poblacion.sort(key=lambda individuo: individuo[0])

    mejor = poblacion[0]
    historial_mejor = [mejor[0]]
    historial_promedio = [sum(individuo[0] for individuo in poblacion) / len(poblacion)]
    generacion = generacion_mejor = sin_mejora = 0

    while generacion < p["generaciones"] and sin_mejora < p["paciencia"]:
        generacion += 1
        nueva = poblacion[:p["elitismo"]]
        vistos = {tuple(individuo[1]) for individuo in nueva}
        for _ in range(tam_poblacion * INTENTOS_POR_INDIVIDUO):
            if len(nueva) == tam_poblacion:
                break
            padre_1 = seleccion_torneo(poblacion, p["tam_torneo"], rng)
            padre_2 = seleccion_torneo(poblacion, p["tam_torneo"], rng)
            if rng.random() < p["prob_cruza"]:
                hijos = (cruza_ox(padre_1, padre_2, rng), cruza_ox(padre_2, padre_1, rng))
            else:
                hijos = (list(padre_1), list(padre_2))
            for hijo in hijos:
                if len(nueva) == tam_poblacion:
                    break
                if rng.random() < p["prob_mutacion"]:
                    hijo = mutar(hijo, rng)
                agregar(nueva, vistos, hijo)
        poblacion = sorted(nueva, key=lambda individuo: individuo[0])

        if poblacion[0][0] < mejor[0] - TOLERANCIA_MEJORA:
            mejor = poblacion[0]
            generacion_mejor = generacion
            sin_mejora = 0
        else:
            sin_mejora += 1
        historial_mejor.append(mejor[0])
        historial_promedio.append(sum(individuo[0] for individuo in poblacion) / len(poblacion))

    costo, rutas = split_rapido(mejor[1], matriz, demandas, capacidad)
    return {
        "rutas": rutas,
        "costo": costo,
        "gran_ruta": list(mejor[1]),
        "historial_mejor": historial_mejor,
        "historial_promedio": historial_promedio,
        "generaciones": generacion,
        "generacion_mejor": generacion_mejor,
        "evaluaciones": evaluaciones,
        "evaluaciones_hasta_mejor": mejor[2],
        "tiempo": time.perf_counter() - inicio,
        "tiempo_hasta_mejor": mejor[3],
        "parametros": p,
    }

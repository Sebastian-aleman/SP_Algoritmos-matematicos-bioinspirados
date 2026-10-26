"""Algoritmo genético de la Etapa 3 (extiende genetico.py de la Entrega 2).

Mismo esquema: cromosoma de gran ruta, split óptimo (Prins, 2004), elitismo, control de
clones y mutación por intercambio, inversión o inserción. Cambios:
- puntos_cruce: 1 (OX de un punto) o 2 (OX de dos puntos, el de la Entrega 2).
- seleccion: "torneo" (tamaño tam_torneo) o "ruleta" (probabilidad proporcional a
  Z_peor - Z, escalamiento usual para minimizar; Goldberg, 1989).
- inicializacion: "vecino" (vecino más cercano aleatorizado, ver _gran_ruta_vecino) o
  "aleatoria" (permutaciones al azar, como en la Entrega 2).
- split con duración máxima de ruta (CMT6-10, 13 y 14).
- criterio de paro: generaciones máximas, paciencia o tiempo máximo (lo que ocurra antes).
- prob_busqueda_local > 0 da la versión memética: con esa probabilidad, las rutas de un
  hijo se mejoran con busqueda_local3 y el cromosoma se re-codifica concatenándolas.
"""

import math
import random
import time

from src.heuristica3 import busqueda_local3
from src.instancias_cvrplib import vecinos_cercanos

PARAMETROS_GA3 = {
    "tam_poblacion": 50,
    "prob_cruza": 0.9,
    "puntos_cruce": 2,
    "prob_mutacion": 0.05,
    "seleccion": "torneo",
    "tam_torneo": 5,
    "elitismo": 5,
    "generaciones": 100000,
    "paciencia": 300,
    "tiempo_max": 60.0,
    "inicializacion": "vecino",
    "vecinos_inicial": 3,
    "prob_busqueda_local": 0.0,
}

TOLERANCIA_MEJORA = 1e-9
INTENTOS_POR_INDIVIDUO = 50


def split_duracion(gran_ruta, matriz, demandas, capacidad, limite=None, servicio=0.0):
    """Partición óptima de la gran ruta en rutas que respetan capacidad y duración. O(n * L)."""
    m = len(gran_ruta)
    V = [math.inf] * (m + 1)
    P = [0] * (m + 1)
    V[0] = 0.0
    fila_deposito = matriz[0]
    for i in range(m):
        vi = V[i]
        if vi == math.inf:
            continue
        primero = gran_ruta[i]
        carga = 0
        tramo = 0.0
        anterior = primero
        for j in range(i, m):
            cliente = gran_ruta[j]
            carga += demandas[cliente]
            if carga > capacidad:
                break
            if j > i:
                tramo += matriz[anterior][cliente]
            anterior = cliente
            d = fila_deposito[primero] + tramo + matriz[cliente][0]
            if limite is not None and d + servicio * (j - i + 1) > limite + 1e-9:
                break
            if vi + d < V[j + 1]:
                V[j + 1] = vi + d
                P[j + 1] = i
    rutas, j = [], m
    while j > 0:
        rutas.append(gran_ruta[P[j]:j])
        j = P[j]
    return V[m], rutas[::-1]


def cruza_ox1(padre_1, padre_2, rng):
    """OX de un punto: conserva el prefijo de padre_1 y completa en el orden de padre_2."""
    n = len(padre_1)
    corte = rng.randrange(1, n)
    prefijo = padre_1[:corte]
    copiados = set(prefijo)
    return prefijo + [c for c in padre_2 if c not in copiados]


def cruza_ox2(padre_1, padre_2, rng):
    """OX de dos puntos (Davis, 1985), igual que en la Entrega 2."""
    n = len(padre_1)
    i, j = sorted(rng.sample(range(n), 2))
    hijo = [None] * n
    hijo[i:j + 1] = padre_1[i:j + 1]
    copiados = set(padre_1[i:j + 1])
    orden_padre_2 = [padre_2[(j + 1 + k) % n] for k in range(n)]
    restantes = [c for c in orden_padre_2 if c not in copiados]
    for k, c in enumerate(restantes):
        hijo[(j + 1 + k) % n] = c
    return hijo


def mutar(cromosoma, rng):
    hijo = list(cromosoma)
    a, b = rng.sample(range(len(hijo)), 2)
    operador = rng.randrange(3)
    if operador == 0:
        hijo[a], hijo[b] = hijo[b], hijo[a]
    elif operador == 1:
        i, j = min(a, b), max(a, b)
        hijo[i:j + 1] = reversed(hijo[i:j + 1])
    else:
        hijo.insert(b, hijo.pop(a))
    return hijo


def _gran_ruta_vecino(vecinos, n, rng, r):
    """Vecino más cercano aleatorizado: empieza en un cliente al azar y en cada paso elige
    al azar entre los r clientes no visitados más cercanos al actual."""
    visitado = [False] * (n + 1)
    actual = rng.randint(1, n)
    visitado[actual] = True
    gran_ruta = [actual]
    for _ in range(n - 1):
        opciones = []
        for j in vecinos[actual]:
            if not visitado[j]:
                opciones.append(j)
                if len(opciones) == r:
                    break
        actual = rng.choice(opciones)
        visitado[actual] = True
        gran_ruta.append(actual)
    return gran_ruta


def algoritmo_genetico3(instancia, semilla=None, vecinos=None, **parametros):
    """GA generacional con elitismo. Devuelve rutas, costo y métricas de convergencia."""
    desconocidos = set(parametros) - set(PARAMETROS_GA3)
    if desconocidos:
        raise TypeError(f"Parámetros desconocidos: {', '.join(sorted(desconocidos))}")
    p = {**PARAMETROS_GA3, **parametros}
    matriz, demandas = instancia["matriz"], instancia["demandas"]
    capacidad, n = instancia["capacidad"], instancia["n"]
    limite, servicio = instancia["limite_ruta"], instancia["tiempo_servicio"]
    tam = p["tam_poblacion"]
    cruza = cruza_ox1 if p["puntos_cruce"] == 1 else cruza_ox2

    inicio = time.perf_counter()
    rng = random.Random(semilla)
    evaluaciones = 0

    def transcurrido():
        return time.perf_counter() - inicio

    def agregar(poblacion, vistos, cromosoma):
        nonlocal evaluaciones
        clave = tuple(cromosoma)
        if clave in vistos:
            return False
        vistos.add(clave)
        costo, rutas = split_duracion(cromosoma, matriz, demandas, capacidad, limite, servicio)
        evaluaciones += 1
        if p["prob_busqueda_local"] > 0 and rng.random() < p["prob_busqueda_local"]:
            rutas, _ = busqueda_local3(rutas, instancia, vecinos)
            cromosoma = [c for ruta in rutas for c in ruta]
            if tuple(cromosoma) in vistos:
                return False
            vistos.add(tuple(cromosoma))
            costo, _ = split_duracion(cromosoma, matriz, demandas, capacidad, limite, servicio)
            evaluaciones += 1
        poblacion.append((costo, cromosoma, evaluaciones, transcurrido()))
        return True

    if vecinos is None and (p["inicializacion"] == "vecino" or p["prob_busqueda_local"] > 0):
        vecinos = vecinos_cercanos(matriz)

    # Población inicial
    if p["inicializacion"] == "vecino":
        generar = lambda: _gran_ruta_vecino(vecinos, n, rng, p["vecinos_inicial"])
    else:
        clientes = list(range(1, n + 1))
        generar = lambda: rng.sample(clientes, n)
    poblacion, vistos = [], set()
    for _ in range(tam * INTENTOS_POR_INDIVIDUO):
        if len(poblacion) == tam:
            break
        agregar(poblacion, vistos, generar())
    poblacion.sort(key=lambda ind: ind[0])

    mejor = poblacion[0]
    traza = [(0, mejor[3], mejor[0])]  # (generación, segundos, costo) en cada mejora
    historial_promedio = [sum(ind[0] for ind in poblacion) / len(poblacion)]
    generacion = generacion_mejor = sin_mejora = 0
    motivo = "generaciones"

    while generacion < p["generaciones"]:
        if sin_mejora >= p["paciencia"]:
            motivo = "paciencia"
            break
        if transcurrido() >= p["tiempo_max"]:
            motivo = "tiempo"
            break
        generacion += 1

        if p["seleccion"] == "ruleta":
            peor = poblacion[-1][0]
            mejor_costo = poblacion[0][0]
            epsilon = 1e-6 * max(peor - mejor_costo, 1e-9)
            acumulado, pesos = 0.0, []
            for ind in poblacion:
                acumulado += peor - ind[0] + epsilon
                pesos.append(acumulado)
            seleccionar = lambda: rng.choices(poblacion, cum_weights=pesos, k=1)[0][1]
        else:
            k = min(p["tam_torneo"], len(poblacion))
            seleccionar = lambda: min(rng.sample(poblacion, k), key=lambda ind: ind[0])[1]

        nueva = poblacion[:p["elitismo"]]
        vistos = {tuple(ind[1]) for ind in nueva}
        for _ in range(tam * INTENTOS_POR_INDIVIDUO):
            if len(nueva) == tam:
                break
            padre_1, padre_2 = seleccionar(), seleccionar()
            if rng.random() < p["prob_cruza"]:
                hijos = (cruza(padre_1, padre_2, rng), cruza(padre_2, padre_1, rng))
            else:
                hijos = (list(padre_1), list(padre_2))
            for hijo in hijos:
                if len(nueva) == tam:
                    break
                if rng.random() < p["prob_mutacion"]:
                    hijo = mutar(hijo, rng)
                agregar(nueva, vistos, hijo)
        poblacion = sorted(nueva, key=lambda ind: ind[0])

        if poblacion[0][0] < mejor[0] - TOLERANCIA_MEJORA:
            mejor = poblacion[0]
            generacion_mejor = generacion
            sin_mejora = 0
            traza.append((generacion, mejor[3], mejor[0]))
        else:
            sin_mejora += 1
        historial_promedio.append(sum(ind[0] for ind in poblacion) / len(poblacion))

    costo, rutas = split_duracion(mejor[1], matriz, demandas, capacidad, limite, servicio)
    return {
        "rutas": rutas,
        "costo": costo,
        "iteraciones": generacion,
        "iteracion_mejor": generacion_mejor,
        "evaluaciones": evaluaciones,
        "evaluaciones_hasta_mejor": mejor[2],
        "tiempo": transcurrido(),
        "tiempo_hasta_mejor": mejor[3],
        "motivo_paro": motivo,
        "traza": traza,
        "costo_inicial": traza[0][2],
        "promedio_inicial": historial_promedio[0],
        "promedio_final": historial_promedio[-1],
        "parametros": p,
    }

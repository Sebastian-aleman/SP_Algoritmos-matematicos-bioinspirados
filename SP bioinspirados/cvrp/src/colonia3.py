"""Colonia de hormigas de la Etapa 3 (extiende colonia_hormigas.py de la Entrega 2).

Mismo algoritmo: Ant System con rango (Bullnheimer, Hartl y Strauss, 1999), visibilidad
por ahorros, lista de candidatos, 2-opt en cada ruta y tau_0 por vecino más cercano.
Cambios para escalar a cientos de clientes:
- num_hormigas = None usa min(n, MAX_HORMIGAS). Con m = n (Entrega 2) una iteración en
  X-n459 tardaría decenas de segundos en Python puro. Si m = n, la hormiga k abre su
  primera ruta en el cliente k (como en la Entrega 2); si m < n, en un cliente al azar.
- Los clientes pendientes se guardan en un arreglo de banderas y una lista con borrado
  en O(1); la lista de candidatos se filtra sin recorrer a todos los clientes.
- Factibilidad con capacidad y duración máxima de ruta (CMT6-10, 13 y 14).
- Criterio de paro: iteraciones máximas, paciencia o tiempo máximo.
"""

import math
import random
import time

from src.colonia_hormigas import visibilidad_ahorros
from src.heuristica3 import dos_opt_ruta
from src.instancias_cvrplib import costo_solucion, vecinos_cercanos

MAX_HORMIGAS = 25

PARAMETROS_ACO3 = {
    "num_hormigas": None,
    "alfa": 5,
    "beta": 5,
    "rho": 0.75,
    "sigma": 6,
    "f": 2,
    "g": 2,
    "fraccion_candidatos": 0.25,
    "dos_opt": True,
    "iteraciones": 100000,
    "paciencia": 100,
    "tiempo_max": 60.0,
}

TOLERANCIA = 1e-9


def vecino_mas_cercano3(instancia):
    """Vecino más cercano con capacidad y duración. Devuelve (costo, rutas)."""
    matriz, demandas, capacidad = instancia["matriz"], instancia["demandas"], instancia["capacidad"]
    limite, servicio = instancia["limite_ruta"], instancia["tiempo_servicio"]
    pendientes = set(range(1, instancia["n"] + 1))
    rutas = []
    while pendientes:
        ruta, carga, longitud, actual = [], 0, 0.0, 0
        while True:
            factibles = [j for j in pendientes if carga + demandas[j] <= capacidad
                         and (limite is None or longitud + matriz[actual][j] + matriz[j][0]
                              + servicio * (len(ruta) + 1) <= limite + 1e-9)]
            if not factibles:
                break
            siguiente = min(factibles, key=lambda j: (matriz[actual][j], j))
            longitud += matriz[actual][siguiente]
            ruta.append(siguiente)
            carga += demandas[siguiente]
            pendientes.remove(siguiente)
            actual = siguiente
        rutas.append(ruta)
    return costo_solucion(rutas, matriz), rutas


def colonia_hormigas3(instancia, semilla=None, vecinos=None, **parametros):
    """Ant System con rango para el CVRP (con duración máxima opcional)."""
    desconocidos = set(parametros) - set(PARAMETROS_ACO3)
    if desconocidos:
        raise TypeError(f"Parámetros desconocidos: {', '.join(sorted(desconocidos))}")
    p = {**PARAMETROS_ACO3, **parametros}
    matriz, demandas = instancia["matriz"], instancia["demandas"]
    capacidad, n = instancia["capacidad"], instancia["n"]
    limite, servicio = instancia["limite_ruta"], instancia["tiempo_servicio"]
    num_hormigas = min(n, MAX_HORMIGAS) if p["num_hormigas"] is None else p["num_hormigas"]
    alfa, rho, sigma = p["alfa"], p["rho"], p["sigma"]

    inicio = time.perf_counter()
    rng = random.Random(semilla)
    N = n + 1
    if vecinos is None:
        vecinos = vecinos_cercanos(matriz)
    tamano = max(1, math.floor(p["fraccion_candidatos"] * n))
    candidatos = [None] + [vecinos[i][:tamano] for i in range(1, N)]
    eta_beta = [[v ** p["beta"] for v in fila]
                for fila in visibilidad_ahorros(matriz, p["f"], p["g"])]
    z_vmc, _ = vecino_mas_cercano3(instancia)
    tau0 = sigma / ((1 - rho) * z_vmc)
    tau = [[tau0] * N for _ in range(N)]

    def transcurrido():
        return time.perf_counter() - inicio

    def construir(primero, pesos):
        pendiente = [False] + [True] * n
        lista = list(range(1, N))
        posicion = list(range(-1, n))  # posicion[c] = índice de c en lista
        restantes = n

        def quitar(c):
            nonlocal restantes
            pendiente[c] = False
            k = posicion[c]
            ultimo = lista[-1]
            lista[k] = ultimo
            posicion[ultimo] = k
            lista.pop()
            restantes -= 1

        rutas = []
        while restantes:
            ruta, carga, longitud, actual = [], 0, 0.0, 0
            while True:
                libre = capacidad - carga
                usados = servicio * (len(ruta) + 1)

                def cabe(j):
                    if demandas[j] > libre:
                        return False
                    return (limite is None or longitud + matriz[actual][j] + matriz[j][0]
                            + usados <= limite + 1e-9)

                if primero is not None:
                    elegido, primero = primero, None
                else:
                    opciones = None
                    if actual != 0:
                        opciones = [j for j in candidatos[actual] if pendiente[j] and cabe(j)]
                    if not opciones:
                        opciones = [j for j in lista if cabe(j)]
                    if not opciones:
                        break
                    fila = pesos[actual]
                    acumulado, umbrales = 0.0, []
                    for j in opciones:
                        acumulado += fila[j]
                        umbrales.append(acumulado)
                    if acumulado > 0 and math.isfinite(acumulado):
                        u = rng.random() * acumulado
                        elegido = opciones[-1]
                        for j, umbral in zip(opciones, umbrales):
                            if umbral >= u:
                                elegido = j
                                break
                    else:
                        elegido = rng.choice(opciones)
                longitud += matriz[actual][elegido]
                ruta.append(elegido)
                carga += demandas[elegido]
                quitar(elegido)
                actual = elegido
            rutas.append(dos_opt_ruta(ruta, matriz) if p["dos_opt"] else ruta)
        return costo_solucion(rutas, matriz), rutas

    mejor = None
    traza = []
    iteracion = iteracion_mejor = sin_mejora = evaluaciones = 0
    promedio_inicial = promedio_final = None
    motivo = "iteraciones"

    while iteracion < p["iteraciones"]:
        if sin_mejora >= p["paciencia"]:
            motivo = "paciencia"
            break
        if iteracion > 0 and transcurrido() >= p["tiempo_max"]:
            motivo = "tiempo"
            break
        iteracion += 1
        pesos = [[tau[i][j] ** alfa * eta_beta[i][j] for j in range(N)] for i in range(N)]
        soluciones = []
        for k in range(num_hormigas):
            primero = k + 1 if num_hormigas == n else rng.randint(1, n)
            costo, rutas = construir(primero, pesos)
            evaluaciones += 1
            soluciones.append((costo, rutas, evaluaciones, transcurrido()))
        soluciones.sort(key=lambda s: s[0])
        promedio_final = sum(s[0] for s in soluciones) / len(soluciones)
        if promedio_inicial is None:
            promedio_inicial = promedio_final

        if mejor is None or soluciones[0][0] < mejor[0] - TOLERANCIA:
            mejor = soluciones[0]
            iteracion_mejor = iteracion
            sin_mejora = 0
            traza.append((iteracion, mejor[3], mejor[0]))
        else:
            sin_mejora += 1

        for fila in tau:
            for j in range(N):
                fila[j] *= rho
        for mu, (costo, rutas, _, _) in enumerate(soluciones[:sigma - 1], start=1):
            _depositar(tau, rutas, (sigma - mu) / costo)
        _depositar(tau, mejor[1], sigma / mejor[0])

    return {
        "rutas": [list(r) for r in mejor[1]],
        "costo": mejor[0],
        "iteraciones": iteracion,
        "iteracion_mejor": iteracion_mejor,
        "evaluaciones": evaluaciones,
        "evaluaciones_hasta_mejor": mejor[2],
        "tiempo": transcurrido(),
        "tiempo_hasta_mejor": mejor[3],
        "motivo_paro": motivo,
        "traza": traza,
        "costo_inicial": traza[0][2],
        "promedio_inicial": promedio_inicial,
        "promedio_final": promedio_final,
        "num_hormigas": num_hormigas,
        "parametros": p,
    }


def _depositar(tau, rutas, cantidad):
    for ruta in rutas:
        anterior = 0
        for c in list(ruta) + [0]:
            tau[anterior][c] += cantidad
            tau[c][anterior] += cantidad
            anterior = c

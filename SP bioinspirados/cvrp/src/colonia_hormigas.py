"""Colonia de hormigas para el CVRP: Ant System con rango (Bullnheimer, Hartl y Strauss, 1999).

Cada hormiga construye una solución factible cliente por cliente y regresa al
depósito solo cuando ningún cliente pendiente cabe en el vehículo. La visibilidad
se basa en los ahorros de Clarke-Wright, la elección se restringe a una lista de
candidatos y la feromona se actualiza con las sigma-1 mejores hormigas de la
iteración más la mejor solución global.
"""

import math
import random
import time

# Valores calibrados en main_calibracion.py; un valor cambia solo si reduce las
# evaluaciones hasta el mejor en más de 2 errores estándar. iteraciones y paciencia
# se recortaron a la convergencia observada (se confirma en main_entrega2.py).
PARAMETROS_ACO = {
    "num_hormigas": None,  # None = n (una hormiga por cliente)
    "iteraciones": 50,
    "alfa": 5,
    "beta": 5,
    "rho": 0.75,  # persistencia de la feromona (la evaporación es 1 - rho)
    "sigma": 6,
    "f": 2,
    "g": 2,
    "fraccion_candidatos": 0.25,
    "dos_opt": True,
    "paciencia": 20,
}

TOLERANCIA = 1e-9
EPSILON_VISIBILIDAD = 1e-6


def visibilidad_ahorros(matriz, f, g):
    """eta_ij = d_i0 + d_0j - g*d_ij + f*|d_i0 - d_0j|; los valores <= 0 pasan a 1e-6."""
    n = len(matriz)
    eta = [[EPSILON_VISIBILIDAD] * n for _ in range(n)]
    for i in range(n):
        for j in range(1, n):
            if i == j:
                continue
            valor = (matriz[i][0] + matriz[0][j] - g * matriz[i][j]
                     + f * abs(matriz[i][0] - matriz[0][j]))
            eta[i][j] = valor if valor > 0 else EPSILON_VISIBILIDAD
    return eta


def listas_candidatos(matriz, fraccion_candidatos):
    """Para cada cliente i, el conjunto de sus floor(fraccion * n) clientes más cercanos."""
    clientes = range(1, len(matriz))
    tamano = math.floor(fraccion_candidatos * len(clientes))
    candidatos = [set()]  # el depósito no usa lista: considera a todos los clientes
    for i in clientes:
        cercanos = sorted((j for j in clientes if j != i), key=lambda j: (matriz[i][j], j))
        candidatos.append(set(cercanos[:tamano]))
    return candidatos


def costo_rutas(rutas, matriz):
    """Distancia total de las rutas (cada una sale y regresa al depósito)."""
    total = 0.0
    for ruta in rutas:
        secuencia = [0] + list(ruta) + [0]
        total += sum(matriz[a][b] for a, b in zip(secuencia, secuencia[1:]))
    return total


def vecino_mas_cercano(matriz, demandas, capacidad):
    """Vecino más cercano con capacidad: va al cliente pendiente más cercano que quepa
    y regresa al depósito cuando ninguno cabe. Devuelve (costo, rutas)."""
    pendientes = list(range(1, len(matriz)))
    rutas = []
    while pendientes:
        ruta, carga, actual = [], 0, 0
        while True:
            factibles = [j for j in pendientes if carga + demandas[j] <= capacidad]
            if not factibles:
                break
            actual = min(factibles, key=lambda j: (matriz[actual][j], j))
            ruta.append(actual)
            carga += demandas[actual]
            pendientes.remove(actual)
        rutas.append(ruta)
    return costo_rutas(rutas, matriz), rutas


def dos_opt(ruta, matriz):
    """2-opt con mejor mejora sobre una ruta, hasta que ningún movimiento la acorte."""
    sec = [0] + list(ruta) + [0]
    while True:
        mejor_delta, mejor_par = -TOLERANCIA, None
        for a in range(1, len(sec) - 2):
            for b in range(a + 1, len(sec) - 1):
                delta = (matriz[sec[a - 1]][sec[b]] + matriz[sec[a]][sec[b + 1]]
                         - matriz[sec[a - 1]][sec[a]] - matriz[sec[b]][sec[b + 1]])
                if delta < mejor_delta:
                    mejor_delta, mejor_par = delta, (a, b)
        if mejor_par is None:
            return sec[1:-1]
        a, b = mejor_par
        sec[a:b + 1] = reversed(sec[a:b + 1])


def _ruleta(opciones, pesos_desde, rng):
    """Elige una opción con probabilidad proporcional a su peso (uniforme si todos son 0)."""
    pesos = [pesos_desde[j] for j in opciones]
    total = sum(pesos)
    if not total > 0 or math.isinf(total):
        return rng.choice(opciones)
    umbral = rng.random() * total
    acumulado = 0.0
    for opcion, peso in zip(opciones, pesos):
        acumulado += peso
        if acumulado >= umbral:
            return opcion
    return opciones[-1]


def _depositar(tau, rutas, cantidad):
    """Suma `cantidad` a tau_ij y tau_ji en cada arco de las rutas."""
    for ruta in rutas:
        secuencia = [0] + list(ruta) + [0]
        for a, b in zip(secuencia, secuencia[1:]):
            tau[a][b] += cantidad
            tau[b][a] += cantidad


def colonia_hormigas(matriz, demandas, capacidad, semilla=None, **parametros):
    """Ant System con rango para el CVRP.

    Los parámetros no indicados toman su valor de PARAMETROS_ACO. Devuelve un
    diccionario con: rutas, costo, historial_mejor e historial_promedio (uno por
    iteración), iteraciones (ejecutadas), iteracion_mejor, evaluaciones
    (soluciones construidas por las hormigas), evaluaciones_hasta_mejor y
    tiempo_hasta_mejor (al construir la mejor solución final), tiempo (s) y
    parametros.
    """
    desconocidos = set(parametros) - set(PARAMETROS_ACO)
    if desconocidos:
        raise TypeError(f"Parámetros desconocidos: {', '.join(sorted(desconocidos))}")
    p = {**PARAMETROS_ACO, **parametros}
    clientes = list(range(1, len(matriz)))
    if any(demandas[j] > capacidad for j in clientes):
        raise ValueError("Hay clientes cuya demanda excede la capacidad del vehículo")
    num_hormigas = len(clientes) if p["num_hormigas"] is None else p["num_hormigas"]

    inicio = time.perf_counter()
    rng = random.Random(semilla)
    n = len(matriz)
    eta_beta = [[valor ** p["beta"] for valor in fila]
                for fila in visibilidad_ahorros(matriz, p["f"], p["g"])]
    candidatos = listas_candidatos(matriz, p["fraccion_candidatos"])
    z_vmc, _ = vecino_mas_cercano(matriz, demandas, capacidad)
    tau0 = p["sigma"] / ((1 - p["rho"]) * z_vmc)
    tau = [[tau0] * n for _ in range(n)]

    def construir(primero, pesos):
        """Solución de una hormiga; `primero` es el cliente con el que abre su primera ruta."""
        pendientes = list(clientes)
        rutas = []
        while pendientes:
            ruta, carga, actual = [], 0, 0
            while True:
                factibles = [j for j in pendientes if carga + demandas[j] <= capacidad]
                if not factibles:
                    break
                if primero is not None:
                    elegido, primero = primero, None
                elif actual == 0:
                    elegido = _ruleta(factibles, pesos[0], rng)
                else:
                    en_lista = [j for j in factibles if j in candidatos[actual]]
                    elegido = _ruleta(en_lista or factibles, pesos[actual], rng)
                ruta.append(elegido)
                carga += demandas[elegido]
                pendientes.remove(elegido)
                actual = elegido
            rutas.append(dos_opt(ruta, matriz) if p["dos_opt"] else ruta)
        return costo_rutas(rutas, matriz), rutas

    mejor = None
    historial_mejor, historial_promedio = [], []
    iteracion = iteracion_mejor = sin_mejora = evaluaciones = 0

    while iteracion < p["iteraciones"] and sin_mejora < p["paciencia"]:
        iteracion += 1
        pesos = [[tau[i][j] ** p["alfa"] * eta_beta[i][j] for j in range(n)] for i in range(n)]
        soluciones = []
        for k in range(num_hormigas):
            primero = clientes[k] if num_hormigas == len(clientes) else rng.choice(clientes)
            costo, rutas = construir(primero, pesos)
            # Solución: (costo, rutas, evaluaciones y segundos al momento de construirla)
            soluciones.append((costo, rutas, evaluaciones + k + 1, time.perf_counter() - inicio))
        evaluaciones += num_hormigas
        soluciones.sort(key=lambda solucion: solucion[0])

        if mejor is None or soluciones[0][0] < mejor[0] - TOLERANCIA:
            mejor = soluciones[0]
            iteracion_mejor = iteracion
            sin_mejora = 0
        else:
            sin_mejora += 1
        historial_mejor.append(mejor[0])
        historial_promedio.append(sum(solucion[0] for solucion in soluciones) / len(soluciones))

        # Actualización por rango: evaporación, sigma-1 mejores hormigas y mejor global
        for fila in tau:
            for j in range(n):
                fila[j] *= p["rho"]
        for mu, (costo, rutas, _, _) in enumerate(soluciones[:p["sigma"] - 1], start=1):
            _depositar(tau, rutas, (p["sigma"] - mu) / costo)
        _depositar(tau, mejor[1], p["sigma"] / mejor[0])

    return {
        "rutas": [list(ruta) for ruta in mejor[1]],
        "costo": mejor[0],
        "historial_mejor": historial_mejor,
        "historial_promedio": historial_promedio,
        "iteraciones": iteracion,
        "iteracion_mejor": iteracion_mejor,
        "evaluaciones": evaluaciones,
        "evaluaciones_hasta_mejor": mejor[2],
        "tiempo": time.perf_counter() - inicio,
        "tiempo_hasta_mejor": mejor[3],
        "parametros": p,
    }

"""Modelo exacto del CVRP (formulación de dos índices con restricciones MTZ de carga)."""

import math
import re
import time

import pulp

LIMITE_TIEMPO = 600  # segundos


def construir_modelo(matriz, demandas, capacidad):
    """Construye el modelo. Devuelve (modelo, x, u)."""
    n = len(matriz)
    nodos = range(n)
    clientes = range(1, n)
    vehiculos_minimos = math.ceil(sum(demandas) / capacidad)

    modelo = pulp.LpProblem("CVRP", pulp.LpMinimize)
    x = {(i, j): modelo.add_variable(f"x_{i}_{j}", cat="Binary")
         for i in nodos for j in nodos if i != j}
    # q[i] <= u[i] <= Q se impone como cota de la variable
    u = {i: modelo.add_variable(f"u_{i}", lowBound=demandas[i], upBound=capacidad)
         for i in clientes}

    modelo += pulp.lpSum(matriz[i][j] * x[i, j] for i, j in x), "distancia_total"

    for i in clientes:
        modelo += pulp.lpSum(x[i, j] for j in nodos if j != i) == 1, f"salida_{i}"
    for j in clientes:
        modelo += pulp.lpSum(x[i, j] for i in nodos if i != j) == 1, f"entrada_{j}"

    salidas_deposito = pulp.lpSum(x[0, j] for j in clientes)
    entradas_deposito = pulp.lpSum(x[i, 0] for i in clientes)
    modelo += salidas_deposito == entradas_deposito, "balance_deposito"
    modelo += salidas_deposito >= vehiculos_minimos, "vehiculos_minimos"

    for i in clientes:
        for j in clientes:
            if i != j:
                modelo += (u[j] >= u[i] + demandas[j] - capacidad * (1 - x[i, j])
                           + (capacidad - demandas[i] - demandas[j]) * x[j, i],
                           f"carga_{i}_{j}")

    return modelo, x, u


def _crear_solver(ruta_log):
    # En PuLP >= 4 CBC se usa con COIN_CMD y el binario del paquete cbcbox
    return pulp.COIN_CMD(msg=False, timeLimit=LIMITE_TIEMPO,
                         logPath=str(ruta_log) if ruta_log else None)


def _leer_log_cbc(ruta_log):
    """Extrae del log de CBC la cota inferior final y la relajación lineal."""
    datos = {"cota_inferior": None, "relajacion_lineal": None}
    try:
        with open(ruta_log, encoding="utf-8", errors="replace") as archivo:
            texto = archivo.read()
    except (OSError, TypeError):
        return datos
    numero = r"(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
    # Línea final del B&C: "+ Optimal - Obj: 318.989   Bound: 318.989   Gap: 0.00% ..."
    cota = re.findall(r"^\+ .*\bBound:\s+" + numero, texto, flags=re.MULTILINE)
    relajacion = re.search(r"LP Optimal .*\bObj:\s+" + numero, texto)
    if cota:
        datos["cota_inferior"] = float(cota[-1])
    if relajacion:
        datos["relajacion_lineal"] = float(relajacion.group(1))
    return datos


def reconstruir_rutas(x, n):
    """Reconstruye las rutas (listas de clientes) siguiendo los arcos con x = 1."""
    siguiente = {}
    inicios = []
    for (i, j), variable in x.items():
        if variable.value() is not None and variable.value() > 0.5:
            if i == 0:
                inicios.append(j)
            else:
                siguiente[i] = j
    rutas = []
    for inicio in sorted(inicios):
        ruta, actual = [], inicio
        while actual != 0:
            ruta.append(actual)
            actual = siguiente[actual]
        rutas.append(ruta)
    if sum(len(r) for r in rutas) != n - 1:
        raise ValueError("La solución contiene subtours que no pasan por el depósito")
    return rutas


def resolver_modelo_exacto(matriz, demandas, capacidad, ruta_log=None):
    """Resuelve el modelo con CBC.

    Devuelve (rutas, resumen) con estatus, Z, cota inferior, relajación lineal,
    número de variables y restricciones, y tiempo de `solve` en segundos.
    """
    modelo, x, u = construir_modelo(matriz, demandas, capacidad)
    solver = _crear_solver(ruta_log)

    inicio = time.perf_counter()
    estadisticas = modelo.solve(solver)
    tiempo = time.perf_counter() - inicio

    estatus = estadisticas.status.name
    z = pulp.value(modelo.objective)
    log = _leer_log_cbc(ruta_log)
    cota = estadisticas.best_bound
    if cota is None:
        cota = log["cota_inferior"]
    if estatus == "Optimal" and (cota is None or abs(cota - z) < 1e-2):
        # Optimalidad probada: la cota coincide con Z (el log la trae con 3 decimales)
        cota = z

    resumen = {
        "estatus": estatus,
        "z": z,
        "cota_inferior": cota,
        "relajacion_lineal": log["relajacion_lineal"],
        "numero_variables": modelo.numVariables(),
        "variables_binarias": len(x),
        "variables_continuas": len(u),
        "numero_restricciones": modelo.numConstraints(),
        "nodos_explorados": getattr(estadisticas, "nodes", None),
        "tiempo": tiempo,
        "solver": f"CBC (PuLP {pulp.__version__})",
    }
    return reconstruir_rutas(x, len(matriz)), resumen

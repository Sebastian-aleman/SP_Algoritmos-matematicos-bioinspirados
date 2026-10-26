"""Modelo exacto de la Entrega 1 (dos índices, MTZ elevado) con límite de tiempo.

Reutiliza construir_modelo y reconstruir_rutas de modelo_exacto.py sin modificarlos.
El modelo no incluye la duración máxima de ruta, así que solo se aplica a instancias
sin DISTANCE (CMT1-5, 11, 12 y subinstancias).
"""

import re
import time

import pulp

from src.modelo_exacto import construir_modelo, reconstruir_rutas


def _leer_log(ruta_log):
    datos = {"resultado": None, "cota_log": None, "gap_log": None, "nodos_log": None}
    try:
        with open(ruta_log, encoding="utf-8", errors="replace") as archivo:
            texto = archivo.read()
    except OSError:
        return datos
    numero = r"(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
    resultado = re.search(r"^Result - (.+)$", texto, flags=re.MULTILINE)
    cota = re.search(r"^Lower bound:\s+" + numero, texto, flags=re.MULTILINE)
    gap = re.search(r"^Gap:\s+" + numero, texto, flags=re.MULTILINE)
    nodos = re.search(r"^Enumerated nodes:\s+(\d+)", texto, flags=re.MULTILINE)
    if resultado:
        datos["resultado"] = resultado.group(1).strip()
    if cota:
        datos["cota_log"] = float(cota.group(1))
    if gap:
        datos["gap_log"] = float(gap.group(1))
    if nodos:
        datos["nodos_log"] = int(nodos.group(1))
    return datos


def resolver_exacto(instancia, limite_tiempo, ruta_log):
    """Resuelve el modelo con CBC. Devuelve un diccionario con estatus, Z, cota y rutas.

    estatus: "optimo" (optimalidad demostrada), "factible" (límite de tiempo con solución
    entera), "sin_solucion" (límite de tiempo sin solución) u otro texto de CBC.
    """
    if instancia["limite_ruta"] is not None:
        raise ValueError("El modelo exacto no incluye la duración máxima de ruta")
    inicio = time.perf_counter()
    modelo, x, _ = construir_modelo(instancia["matriz"], instancia["demandas"],
                                    instancia["capacidad"])
    tiempo_construccion = time.perf_counter() - inicio
    solver = pulp.COIN_CMD(msg=False, timeLimit=int(limite_tiempo), gapRel=0,
                           logPath=str(ruta_log))
    inicio = time.perf_counter()
    estadisticas = modelo.solve(solver)
    tiempo = time.perf_counter() - inicio
    log = _leer_log(ruta_log)

    estatus_pulp = estadisticas.status.name
    z = pulp.value(modelo.objective)
    tiene_solucion = bool(getattr(estadisticas, "has_solution", z is not None)) and z is not None
    gap_cbc = getattr(estadisticas, "gap_rel", None)
    cerrado = (log["resultado"] or "").startswith("Optimal solution found")
    if estatus_pulp == "Optimal" or (tiene_solucion and cerrado
                                     and (gap_cbc is None or gap_cbc <= 1e-9)):
        estatus = "optimo"
    elif tiene_solucion:
        estatus = "factible"
    elif estatus_pulp == "TimeLimit" or (log["resultado"] or "").startswith("Stopped"):
        estatus = "sin_solucion"
    else:
        estatus = estatus_pulp.lower()

    rutas = None
    if tiene_solucion:
        try:
            rutas = reconstruir_rutas(x, len(instancia["matriz"]))
        except (ValueError, KeyError):
            rutas = None
    cota = getattr(estadisticas, "best_bound", None)
    if cota is None:
        cota = log["cota_log"]
    if estatus == "optimo":
        cota = z
    return {
        "estatus": estatus,
        "estatus_cbc": log["resultado"] or estatus_pulp,
        "z": z if tiene_solucion else None,
        "cota_inferior": cota,
        "gap_mip_pct": (None if not tiene_solucion or not cota
                        else (z - cota) / z * 100),
        "nodos": getattr(estadisticas, "nodes", None) or log["nodos_log"],
        "numero_variables": modelo.numVariables(),
        "numero_restricciones": modelo.numConstraints(),
        "tiempo_construccion": tiempo_construccion,
        "tiempo": tiempo,
        "limite_tiempo": limite_tiempo,
        "rutas": rutas,
    }

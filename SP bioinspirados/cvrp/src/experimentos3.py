"""Experimentos de la Etapa 3: definición de tareas, ejecución en paralelo y persistencia.

Cada corrida se guarda como una línea de resultados/etapa3/corridas.jsonl en cuanto
termina, así que el experimento se puede interrumpir y reanudar: al volver a ejecutar,
las corridas ya guardadas se omiten.
"""

import csv
import itertools
import json
import multiprocessing as mp
import statistics
import time
from datetime import datetime
from pathlib import Path

from src.colonia3 import PARAMETROS_ACO3, colonia_hormigas3
from src.genetico3 import PARAMETROS_GA3, algoritmo_genetico3
from src.heuristica3 import busqueda_local3, clarke_wright3
from src.instancias_cvrplib import INSTANCIAS, cargar, validar, vecinos_cercanos

# --- Diseño experimental ---

FACTORES_GA = {
    "puntos_cruce": [1, 2],
    "prob_mutacion": [0.05, 0.2, 0.5],
    "seleccion": ["torneo", "ruleta"],
}
FACTORES_ACO = {
    "alfa": [1, 5],
    "beta": [2, 5],
    "rho": [0.5, 0.9],
}
FIJOS_GA = {"tam_poblacion": 50, "prob_cruza": 0.9, "tam_torneo": 5, "elitismo": 5,
            "paciencia": 300, "inicializacion": "vecino"}
FIJOS_ACO = {"sigma": 6, "f": 2, "g": 2, "fraccion_candidatos": 0.25, "dos_opt": True,
             "num_hormigas": None, "paciencia": 100}

SEMILLAS_FACTORIAL = [0, 1, 2]
SEMILLAS_FINAL = [100, 101, 102, 103, 104]
SUBINSTANCIAS = [f"CMT1-k{k}" for k in (10, 15, 20, 25, 30, 40)]
EXACTO_COMPLETAS = ["CMT1", "CMT2", "CMT3", "CMT12", "CMT11", "CMT4", "CMT5"]
LIMITE_EXACTO = 600
PROB_BL_MEMETICO = 0.1
INSTANCIAS_INICIALIZACION = ["CMT1", "CMT5", "CMT13", "X-n280-k17", "X-n459-k26"]


def tiempo_max(n, escala=1.0):
    """Presupuesto de tiempo por corrida: 0.2 s por cliente, entre 15 y 90 s."""
    return escala * min(max(0.2 * n, 15.0), 90.0)


def _texto(valor):
    return f"{valor:g}" if isinstance(valor, float) else str(valor)


def id_config(algoritmo, niveles):
    if algoritmo == "GA":
        return (f"GA-c{niveles['puntos_cruce']}-m{_texto(niveles['prob_mutacion'])}-"
                f"{niveles['seleccion'][0].upper()}")
    return (f"ACO-a{niveles['alfa']}-b{niveles['beta']}-r{_texto(niveles['rho'])}")


def configuraciones(algoritmo):
    factores = FACTORES_GA if algoritmo == "GA" else FACTORES_ACO
    nombres = list(factores)
    for valores in itertools.product(*(factores[f] for f in nombres)):
        niveles = dict(zip(nombres, valores))
        yield id_config(algoritmo, niveles), niveles


# --- Ejecución de una tarea (corre en un proceso trabajador) ---

_CACHE = {}


def _instancia(nombre, carpeta):
    if nombre not in _CACHE:
        instancia = cargar(nombre, carpeta)
        _CACHE[nombre] = (instancia, vecinos_cercanos(instancia["matriz"]))
    return _CACHE[nombre]


def _metricas_solucion(rutas, instancia):
    valida, mensaje, costo = validar(rutas, instancia)
    numero_rutas = len(rutas)
    capacidad_flota = numero_rutas * instancia["capacidad"]
    return {
        "valida": valida,
        "mensaje_validacion": mensaje,
        "costo_validado": costo,
        "numero_rutas": numero_rutas,
        "desperdicio_global_pct": (capacidad_flota - sum(instancia["demandas"]))
                                  / capacidad_flota * 100,
    }


def ejecutar_tarea(tarea):
    """Ejecuta una tarea y devuelve su registro (diccionario serializable)."""
    instancia, vecinos = _instancia(tarea["instancia"], tarea["carpeta"])
    registro = {k: tarea[k] for k in ("fase", "algoritmo", "config", "instancia", "semilla")}
    registro.update({"n": instancia["n"], "bks": instancia["bks"],
                     "limite_ruta": instancia["limite_ruta"],
                     "niveles": tarea.get("niveles", {})})
    algoritmo = tarea["algoritmo"]

    if algoritmo in ("GA", "ACO"):
        funcion = algoritmo_genetico3 if algoritmo == "GA" else colonia_hormigas3
        r = funcion(instancia, semilla=tarea["semilla"], vecinos=vecinos,
                    tiempo_max=tarea["tiempo_max"], **tarea["parametros"])
        registro.update({k: r[k] for k in (
            "costo", "tiempo", "tiempo_hasta_mejor", "iteraciones", "iteracion_mejor",
            "evaluaciones", "evaluaciones_hasta_mejor", "motivo_paro", "costo_inicial",
            "promedio_inicial", "promedio_final", "traza", "rutas", "parametros")})
        registro["tiempo_max"] = tarea["tiempo_max"]

    elif algoritmo in ("CW", "CW+BL"):
        tiempos = []
        for _ in range(tarea.get("repeticiones", 3)):
            inicio = time.perf_counter()
            rutas = clarke_wright3(instancia)
            movimientos = 0
            if algoritmo == "CW+BL":
                rutas, movimientos = busqueda_local3(rutas, instancia, vecinos_cercanos(
                    instancia["matriz"]))
            tiempos.append(time.perf_counter() - inicio)
        registro.update({"costo": validar(rutas, instancia)[2], "rutas": rutas,
                         "tiempo": statistics.median(tiempos), "iteraciones": movimientos,
                         "repeticiones": len(tiempos)})

    elif algoritmo == "Exacto":
        from src.exacto3 import resolver_exacto
        carpeta_logs = Path(tarea["carpeta_logs"])
        carpeta_logs.mkdir(parents=True, exist_ok=True)
        r = resolver_exacto(instancia, tarea["limite_tiempo"],
                            carpeta_logs / f"cbc_{tarea['instancia']}.log")
        registro.update(r)
        registro["costo"] = r["z"]
    else:
        raise ValueError(f"Algoritmo desconocido: {algoritmo}")

    if registro.get("rutas"):
        registro.update(_metricas_solucion(registro["rutas"], instancia))
    if registro.get("costo") is not None and instancia["bks"]:
        registro["gap_bks_pct"] = (registro["costo"] - instancia["bks"]) / instancia["bks"] * 100
    registro["fecha"] = datetime.now().isoformat(timespec="seconds")
    return registro


def _envoltura(tarea):
    try:
        return ejecutar_tarea(tarea)
    except Exception as error:  # se registra para no perder el resto del experimento
        return {**{k: tarea[k] for k in ("fase", "algoritmo", "config", "instancia", "semilla")},
                "error": repr(error)}


# --- Persistencia ---

def clave(registro):
    return (registro["fase"], registro["algoritmo"], registro["config"],
            registro["instancia"], registro["semilla"])


def leer_corridas(ruta):
    ruta = Path(ruta)
    if not ruta.exists():
        return []
    with open(ruta, encoding="utf-8") as archivo:
        return [json.loads(linea) for linea in archivo if linea.strip()]


def ejecutar_tareas(tareas, ruta_jsonl, trabajadores):
    """Corre las tareas pendientes en paralelo y agrega cada resultado al archivo .jsonl."""
    ruta_jsonl = Path(ruta_jsonl)
    ruta_jsonl.parent.mkdir(parents=True, exist_ok=True)
    hechas = {clave(r) for r in leer_corridas(ruta_jsonl) if "error" not in r}
    pendientes = [t for t in tareas if clave(t) not in hechas]
    # Las tareas largas primero: reparten mejor la carga entre procesos
    pendientes.sort(key=lambda t: -t.get("limite_tiempo", t.get("tiempo_max", 0)))
    print(f"Tareas: {len(tareas)}   ya guardadas: {len(tareas) - len(pendientes)}   "
          f"pendientes: {len(pendientes)}   procesos: {trabajadores}")
    if not pendientes:
        return
    inicio = time.perf_counter()
    with mp.Pool(trabajadores) as pool, open(ruta_jsonl, "a", encoding="utf-8") as archivo:
        for k, registro in enumerate(pool.imap_unordered(_envoltura, pendientes), start=1):
            archivo.write(json.dumps(registro, ensure_ascii=False) + "\n")
            archivo.flush()
            transcurrido = time.perf_counter() - inicio
            restante = transcurrido / k * (len(pendientes) - k)
            if "error" in registro:
                detalle = f"ERROR {registro['error']}"
            else:
                costo = registro.get("costo")
                gap = registro.get("gap_bks_pct")
                detalle = (f"Z={costo:.2f}" if costo is not None else "Z=--")
                detalle += f"  GAP={gap:.2f}%" if gap is not None else ""
                detalle += f"  t={registro.get('tiempo', 0):.1f}s"
                if registro.get("valida") is False:
                    detalle += "  ¡SOLUCIÓN INVÁLIDA!"
            print(f"[{k}/{len(pendientes)}] {registro['algoritmo']:<7} {registro['config']:<22} "
                  f"{registro['instancia']:<11} s={registro['semilla']:<4} {detalle}   "
                  f"(transcurrido {transcurrido / 60:.1f} min, restante ~{restante / 60:.0f} min)",
                  flush=True)


def exportar_csv(ruta_jsonl, ruta_csv):
    """Versión tabular de corridas.jsonl, sin rutas ni trazas."""
    registros = [r for r in leer_corridas(ruta_jsonl) if "error" not in r]
    omitir = {"rutas", "traza", "parametros", "niveles"}
    columnas = []
    for r in registros:
        for c in list(r) + [f"nivel_{k}" for k in r.get("niveles", {})]:
            if c not in omitir and c not in columnas:
                columnas.append(c)
    with open(ruta_csv, "w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=columnas, extrasaction="ignore")
        escritor.writeheader()
        for r in registros:
            fila = dict(r)
            for k, v in r.get("niveles", {}).items():
                fila[f"nivel_{k}"] = v
            escritor.writerow(fila)


# --- Construcción de las tareas de cada fase ---

def tareas_propuesto(carpeta, instancias):
    return [{"fase": "propuesto", "algoritmo": alg, "config": alg, "instancia": inst,
             "semilla": 0, "carpeta": carpeta, "repeticiones": 3}
            for inst in instancias for alg in ("CW", "CW+BL")]


def _tamanos(carpeta, instancias):
    return {inst: cargar(inst, carpeta)["n"] for inst in instancias}


def tareas_factorial(carpeta, instancias, semillas, escala):
    tamanos = _tamanos(carpeta, instancias)
    tareas = []
    for algoritmo, fijos in (("GA", FIJOS_GA), ("ACO", FIJOS_ACO)):
        for config, niveles in configuraciones(algoritmo):
            for inst in instancias:
                n = tamanos[inst]
                for s in semillas:
                    tareas.append({"fase": "factorial", "algoritmo": algoritmo,
                                   "config": config, "niveles": niveles,
                                   "parametros": {**fijos, **niveles}, "instancia": inst,
                                   "semilla": s, "carpeta": carpeta,
                                   "tiempo_max": tiempo_max(n, escala)})
    return tareas


def tareas_exacto(carpeta, instancias, limite, carpeta_logs):
    return [{"fase": "exacto", "algoritmo": "Exacto", "config": "MTZ", "instancia": inst,
             "semilla": 0, "carpeta": carpeta, "limite_tiempo": limite,
             "carpeta_logs": str(carpeta_logs)} for inst in instancias]


def tareas_final(carpeta, instancias, semillas, escala, mejores, instancias_inicializacion,
                 semillas_inicializacion):
    variantes = [
        ("GA", "GA-final", {**FIJOS_GA, **mejores["GA"]}, mejores["GA"]),
        ("ACO", "ACO-final", {**FIJOS_ACO, **mejores["ACO"]}, mejores["ACO"]),
        ("GA", "GA-memetico", {**FIJOS_GA, **mejores["GA"],
                               "prob_busqueda_local": PROB_BL_MEMETICO}, mejores["GA"]),
    ]
    tamanos = _tamanos(carpeta, set(instancias) | set(instancias_inicializacion))
    tareas = []
    for algoritmo, config, parametros, niveles in variantes:
        for inst in instancias:
            n = tamanos[inst]
            for s in semillas:
                tareas.append({"fase": "final", "algoritmo": algoritmo, "config": config,
                               "niveles": niveles, "parametros": parametros,
                               "instancia": inst, "semilla": s, "carpeta": carpeta,
                               "tiempo_max": tiempo_max(n, escala)})
    # Experimento complementario: población inicial aleatoria (como en la Entrega 2)
    for inst in instancias_inicializacion:
        n = tamanos[inst]
        for s in semillas_inicializacion:
            tareas.append({"fase": "final", "algoritmo": "GA", "config": "GA-final-aleatoria",
                           "niveles": mejores["GA"],
                           "parametros": {**FIJOS_GA, **mejores["GA"],
                                          "inicializacion": "aleatoria"},
                           "instancia": inst, "semilla": s, "carpeta": carpeta,
                           "tiempo_max": tiempo_max(n, escala)})
    return tareas


# --- Selección de los mejores niveles (efectos principales por rango promedio) ---

def _rangos(valores):
    """Rangos promedio (1 = menor) con empates repartidos."""
    orden = sorted(range(len(valores)), key=lambda k: valores[k])
    rangos = [0.0] * len(valores)
    k = 0
    while k < len(orden):
        fin = k
        while fin + 1 < len(orden) and abs(valores[orden[fin + 1]] - valores[orden[k]]) < 1e-9:
            fin += 1
        for m in range(k, fin + 1):
            rangos[orden[m]] = (k + fin) / 2 + 1
        k = fin + 1
    return rangos


def seleccionar_mejores(registros, algoritmo):
    """Mejor nivel de cada factor según el rango promedio de las configuraciones que lo usan.

    En cada instancia se ordenan las configuraciones por su costo promedio (rango 1 = mejor);
    el efecto de un nivel es el promedio de los rangos de las configuraciones con ese nivel,
    sobre todas las instancias. Así cada instancia pesa lo mismo sin importar su escala.
    Devuelve (mejores niveles, tabla de efectos, rango promedio por configuración).
    """
    factores = FACTORES_GA if algoritmo == "GA" else FACTORES_ACO
    filas = [r for r in registros if r.get("fase") == "factorial"
             and r.get("algoritmo") == algoritmo and "error" not in r]
    configs = {cid: niv for cid, niv in configuraciones(algoritmo)}
    costos = {}
    for r in filas:
        costos.setdefault((r["config"], r["instancia"]), []).append(r["costo"])
    instancias = sorted({inst for (_, inst) in costos
                         if all((c, inst) in costos for c in configs)})
    if not instancias:
        raise RuntimeError(f"No hay instancias completas del factorial de {algoritmo}")
    rango_config = {c: [] for c in configs}
    for inst in instancias:
        ids = list(configs)
        promedios = [statistics.mean(costos[c, inst]) for c in ids]
        for c, rango in zip(ids, _rangos(promedios)):
            rango_config[c].append(rango)
    rango_promedio = {c: statistics.mean(v) for c, v in rango_config.items()}
    efectos, mejores = {}, {}
    for factor, niveles in factores.items():
        efectos[factor] = {}
        for nivel in niveles:
            usados = [rango_promedio[c] for c, niv in configs.items() if niv[factor] == nivel]
            efectos[factor][nivel] = statistics.mean(usados)
        mejores[factor] = min(niveles, key=lambda nivel: efectos[factor][nivel])
    return mejores, efectos, rango_promedio, instancias

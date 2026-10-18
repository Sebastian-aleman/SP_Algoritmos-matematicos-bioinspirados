"""Entrega 2: confirmación de parámetros y validación del GA y del ACO sobre CVRP_10."""

import csv
import json
import statistics
from pathlib import Path

from main_calibracion import BASE_ACO, BASE_GA
from src.colonia_hormigas import PARAMETROS_ACO, colonia_hormigas
from src.genetico import PARAMETROS_GA, algoritmo_genetico
from src.graficas import graficar_rutas
from src.graficas_bio import graficar_convergencia
from src.instancia import leer_instancia
from src.matriz_costos import calcular_matriz
from src.reporte import evaluar_solucion, guardar_solucion_json, guardar_solucion_tex

RAIZ = Path(__file__).resolve().parent
DATOS = RAIZ / "datos"
RESULTADOS = RAIZ / "resultados"
FIGURAS = RAIZ / "figuras"
SEMILLAS = range(30)
SEMILLAS_VERIFICACION = range(100)
TOLERANCIA_EXITO = 1e-4
TOLERANCIA_EMPATE = 1e-9

# Ganadores individuales de la calibración, juntos (con la población de 20)
COMBINADA_GA = {**BASE_GA, "tam_poblacion": 20, "prob_mutacion": 0.05, "tam_torneo": 5,
                "elitismo": 5, "generaciones": 100, "paciencia": 50}

ALGORITMOS = [
    {"clave": "ga", "nombre": "Algoritmo genético", "siglas": "GA",
     "funcion": algoritmo_genetico,
     "configuraciones": [("BASE", BASE_GA), ("COMBINADA", COMBINADA_GA),
                         ("FINAL", PARAMETROS_GA)],
     "clave_mejor": "generacion_mejor", "etiqueta_mejor": "gen.", "eje_x": "Generación",
     "inicio": 0},  # historial[0] = población inicial
    {"clave": "aco", "nombre": "Colonia de hormigas", "siglas": "ACO",
     "funcion": colonia_hormigas,
     "configuraciones": [("BASE", BASE_ACO), ("FINAL", PARAMETROS_ACO)],
     "clave_mejor": "iteracion_mejor", "etiqueta_mejor": "it.", "eje_x": "Iteración",
     "inicio": 1},  # historial[0] = iteración 1
]

COLUMNAS_CONFIRMACION = [
    "algoritmo", "configuracion", "exito_pct", "z_promedio", "z_maximo",
    "mejor_promedio", "mejor_desviacion", "mejor_maximo",
    "evaluaciones_hasta_mejor_promedio", "evaluaciones_hasta_mejor_desviacion",
    "evaluaciones_promedio", "tiempo_hasta_mejor_mediana", "tiempo_mediana", "parametros"]

COLUMNAS_VERIFICACION = [
    "algoritmo", "semillas", "exito_pct", "z_promedio", "z_maximo", "mejor_promedio",
    "mejor_maximo", "evaluaciones_hasta_mejor_promedio", "tiempo_mediana"]


def titulo(texto):
    print(f"\n{'=' * 78}\n{texto}\n{'=' * 78}")


def imprimir_rutas(solucion):
    print(f"{'ruta':>4}  {'secuencia':<16} {'carga':>5} {'distancia':>10} {'% desperdicio':>14}")
    for r in solucion["rutas"]:
        print(f"{r['ruta']:>4}  {'-'.join(map(str, r['secuencia'])):<16} {r['carga']:>5} "
              f"{r['distancia']:>10.2f} {r['desperdicio_pct']:>14.2f}")
    print(f"Z = {solucion['distancia_total']:.4f}   rutas = {solucion['numero_rutas']}   "
          f"demanda total = {solucion['demanda_total']}")


def es_exito(costo, z_opt):
    return abs(costo - z_opt) < TOLERANCIA_EXITO


def correr(algoritmo, parametros, matriz, demandas, capacidad, semillas=SEMILLAS):
    return [algoritmo["funcion"](matriz, demandas, capacidad, semilla=s, **parametros)
            for s in semillas]


def imprimir_fallas(etiqueta, algoritmo, semillas, corridas, z_opt):
    """Imprime las semillas que no llegan a Z*, con su costo y la generación/iteración del mejor."""
    fallas = [(s, c) for s, c in zip(semillas, corridas) if not es_exito(c["costo"], z_opt)]
    if not fallas:
        print(f"{etiqueta}: todas las semillas ({len(corridas)}) llegan a Z*.")
        return
    print(f"{etiqueta}: {len(fallas)} de {len(corridas)} semillas no llegan a Z*:")
    for semilla, c in fallas:
        print(f"   semilla {semilla:>3}: Z = {c['costo']:.4f}   "
              f"{algoritmo['etiqueta_mejor']} del mejor = {c[algoritmo['clave_mejor']]}")


def forma_canonica(rutas):
    """Solución sin importar el orden de las rutas ni su sentido de recorrido."""
    return tuple(sorted(min(tuple(r), tuple(reversed(r))) for r in rutas))


# --- C1. Confirmación de parámetros ---

def resumir_confirmacion(algoritmo, configuracion, corridas, z_opt):
    """Métricas de las corridas de una configuración (desviaciones muestrales)."""
    costos = [c["costo"] for c in corridas]
    mejores = [c[algoritmo["clave_mejor"]] for c in corridas]
    evaluaciones = [c["evaluaciones_hasta_mejor"] for c in corridas]
    return {
        "algoritmo": algoritmo["siglas"],
        "configuracion": configuracion,
        "exito_pct": sum(es_exito(z, z_opt) for z in costos) / len(costos) * 100,
        "z_promedio": statistics.mean(costos),
        "z_maximo": max(costos),
        "mejor_promedio": statistics.mean(mejores),
        "mejor_desviacion": statistics.stdev(mejores),
        "mejor_maximo": max(mejores),
        "evaluaciones_hasta_mejor_promedio": statistics.mean(evaluaciones),
        "evaluaciones_hasta_mejor_desviacion": statistics.stdev(evaluaciones),
        "evaluaciones_promedio": statistics.mean(c["evaluaciones"] for c in corridas),
        "tiempo_hasta_mejor_mediana": statistics.median(c["tiempo_hasta_mejor"] for c in corridas),
        "tiempo_mediana": statistics.median(c["tiempo"] for c in corridas),
        "parametros": json.dumps(corridas[0]["parametros"]),
    }


def guardar_confirmacion_csv(filas, ruta):
    with open(ruta, "w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=COLUMNAS_CONFIRMACION)
        escritor.writeheader()
        escritor.writerows(filas)


def guardar_confirmacion_tex(filas, ruta):
    lineas = [
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrrrrrrr}",
        r"\toprule",
        r"Algoritmo & Configuración & \% éxito & $Z$ prom. & $Z$ máx. & "
        r"Gen./It. mejor (prom. $\pm$ desv., máx.) & Eval. hasta mejor (prom. $\pm$ desv.) & "
        r"Eval. totales (prom.) & Tiempo hasta mejor (ms, mediana) & "
        r"Tiempo total (ms, mediana) \\",
    ]
    anterior = None
    for f in filas:
        nombre = ""
        if f["algoritmo"] != anterior:
            lineas.append(r"\midrule")
            nombre = f["algoritmo"]
            anterior = f["algoritmo"]
        lineas.append(
            f"{nombre} & {f['configuracion']} & {f['exito_pct']:.1f} & {f['z_promedio']:.2f} & "
            f"{f['z_maximo']:.2f} & {f['mejor_promedio']:.2f} $\\pm$ {f['mejor_desviacion']:.2f} "
            f"({f['mejor_maximo']}) & {f['evaluaciones_hasta_mejor_promedio']:.1f} $\\pm$ "
            f"{f['evaluaciones_hasta_mejor_desviacion']:.1f} & {f['evaluaciones_promedio']:.1f} & "
            f"{f['tiempo_hasta_mejor_mediana'] * 1e3:.2f} & {f['tiempo_mediana'] * 1e3:.2f} \\\\"
        )
    lineas += [
        r"\midrule",
        rf"\multicolumn{{10}}{{l}}{{BASE: valores de la calibración; COMBINADA: ganadores "
        rf"individuales juntos (población 20); FINAL: valores por defecto. {len(SEMILLAS)} "
        rf"semillas; éxito: $|Z - Z^*| < 10^{{-4}}$.}} \\",
        r"\bottomrule",
        r"\end{tabular}%",
        "}",
    ]
    with open(ruta, "w", encoding="utf-8") as archivo:
        archivo.write("\n".join(lineas) + "\n")


def imprimir_confirmacion(filas):
    print(f"{'alg.':<4} {'config.':<9} {'% éxito':>7} {'Z prom.':>9} {'Z máx.':>8} "
          f"{'gen./it. mejor (máx.)':>22} {'eval. hasta mejor':>18} {'eval. totales':>13} "
          f"{'t mejor (ms)':>12} {'t total (ms)':>12}")
    for f in filas:
        mejor = f"{f['mejor_promedio']:.2f} ± {f['mejor_desviacion']:.2f} ({f['mejor_maximo']})"
        evaluaciones = (f"{f['evaluaciones_hasta_mejor_promedio']:.1f} ± "
                        f"{f['evaluaciones_hasta_mejor_desviacion']:.1f}")
        print(f"{f['algoritmo']:<4} {f['configuracion']:<9} {f['exito_pct']:>7.1f} "
              f"{f['z_promedio']:>9.4f} {f['z_maximo']:>8.2f} {mejor:>22} {evaluaciones:>18} "
              f"{f['evaluaciones_promedio']:>13.1f} {f['tiempo_hasta_mejor_mediana'] * 1e3:>12.2f} "
              f"{f['tiempo_mediana'] * 1e3:>12.2f}")


# --- Verificación ampliada (100 semillas) ---

def resumir_verificacion(algoritmo, corridas, z_opt):
    costos = [c["costo"] for c in corridas]
    mejores = [c[algoritmo["clave_mejor"]] for c in corridas]
    return {
        "algoritmo": algoritmo["siglas"],
        "semillas": len(corridas),
        "exito_pct": sum(es_exito(z, z_opt) for z in costos) / len(costos) * 100,
        "z_promedio": statistics.mean(costos),
        "z_maximo": max(costos),
        "mejor_promedio": statistics.mean(mejores),
        "mejor_maximo": max(mejores),
        "evaluaciones_hasta_mejor_promedio": statistics.mean(
            c["evaluaciones_hasta_mejor"] for c in corridas),
        "tiempo_mediana": statistics.median(c["tiempo"] for c in corridas),
    }


def guardar_verificacion_csv(filas, ruta):
    with open(ruta, "w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=COLUMNAS_VERIFICACION)
        escritor.writeheader()
        escritor.writerows(filas)


def guardar_verificacion_tex(filas, ruta):
    lineas = [
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Algoritmo & Semillas & \% éxito & $Z$ prom. & $Z$ máx. & "
        r"Gen./It. mejor (prom., máx.) & Eval. hasta mejor (prom.) & "
        r"Tiempo total (ms, mediana) \\",
        r"\midrule",
    ]
    for f in filas:
        lineas.append(
            f"{f['algoritmo']} & {f['semillas']} & {f['exito_pct']:.1f} & {f['z_promedio']:.2f} & "
            f"{f['z_maximo']:.2f} & {f['mejor_promedio']:.2f} ({f['mejor_maximo']}) & "
            f"{f['evaluaciones_hasta_mejor_promedio']:.1f} & {f['tiempo_mediana'] * 1e3:.2f} \\\\"
        )
    lineas += [
        r"\midrule",
        rf"\multicolumn{{8}}{{l}}{{Configuración FINAL, semillas "
        rf"{SEMILLAS_VERIFICACION.start}--{SEMILLAS_VERIFICACION.stop - 1}; "
        rf"éxito: $|Z - Z^*| < 10^{{-4}}$.}} \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    with open(ruta, "w", encoding="utf-8") as archivo:
        archivo.write("\n".join(lineas) + "\n")


def imprimir_verificacion(filas):
    print(f"{'alg.':<4} {'semillas':>8} {'% éxito':>7} {'Z prom.':>9} {'Z máx.':>8} "
          f"{'gen./it. mejor (máx.)':>22} {'eval. hasta mejor':>17} {'t total (ms)':>12}")
    for f in filas:
        mejor = f"{f['mejor_promedio']:.2f} ({f['mejor_maximo']})"
        print(f"{f['algoritmo']:<4} {f['semillas']:>8} {f['exito_pct']:>7.1f} "
              f"{f['z_promedio']:>9.4f} {f['z_maximo']:>8.2f} {mejor:>22} "
              f"{f['evaluaciones_hasta_mejor_promedio']:>17.1f} {f['tiempo_mediana'] * 1e3:>12.2f}")


# --- C2. Validación de la configuración final ---

def resumir_validacion(algoritmo, corridas, z_opt):
    costos = [c["costo"] for c in corridas]
    numero_rutas = sorted({len(c["rutas"]) for c in corridas})
    return {
        "algoritmo": algoritmo["siglas"],
        "z_mejor": min(costos),
        "z_promedio": statistics.mean(costos),
        "z_peor": max(costos),
        "z_desviacion": statistics.stdev(costos),
        "exito_pct": sum(es_exito(z, z_opt) for z in costos) / len(costos) * 100,
        "soluciones_distintas": len({forma_canonica(c["rutas"]) for c in corridas}),
        "numero_rutas": ", ".join(map(str, numero_rutas)),
        "tiempo_mediana": statistics.median(c["tiempo"] for c in corridas),
    }


def guardar_validacion_tex(filas, ruta):
    lineas = [
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"Algoritmo & $Z$ mejor & $Z$ prom. & $Z$ peor & Desv. & \% éxito & "
        r"Soluciones distintas & N.º rutas & Tiempo total (ms, mediana) \\",
        r"\midrule",
    ]
    for f in filas:
        lineas.append(
            f"{f['algoritmo']} & {f['z_mejor']:.2f} & {f['z_promedio']:.2f} & {f['z_peor']:.2f} & "
            f"{f['z_desviacion']:.2f} & {f['exito_pct']:.1f} & {f['soluciones_distintas']} & "
            f"{f['numero_rutas']} & {f['tiempo_mediana'] * 1e3:.2f} \\\\"
        )
    lineas += [
        r"\midrule",
        rf"\multicolumn{{9}}{{l}}{{Configuración FINAL, {len(SEMILLAS)} semillas. Soluciones "
        rf"distintas: sin importar el orden ni el sentido de las rutas.}} \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    with open(ruta, "w", encoding="utf-8") as archivo:
        archivo.write("\n".join(lineas) + "\n")


def imprimir_validacion(filas):
    print(f"{'alg.':<4} {'Z mejor':>9} {'Z prom.':>9} {'Z peor':>9} {'desv.':>9} {'% éxito':>7} "
          f"{'sol. distintas':>14} {'n.º rutas':>9} {'t total (ms)':>12}")
    for f in filas:
        print(f"{f['algoritmo']:<4} {f['z_mejor']:>9.4f} {f['z_promedio']:>9.4f} "
              f"{f['z_peor']:>9.4f} {f['z_desviacion']:>9.2e} {f['exito_pct']:>7.1f} "
              f"{f['soluciones_distintas']:>14} {f['numero_rutas']:>9} "
              f"{f['tiempo_mediana'] * 1e3:>12.2f}")


def mejor_corrida(corridas):
    """Índice (= semilla) de la corrida de menor costo; en empate, la de menor semilla."""
    z_min = min(c["costo"] for c in corridas)
    return next(s for s, c in enumerate(corridas) if c["costo"] <= z_min + TOLERANCIA_EMPATE)


def main():
    RESULTADOS.mkdir(exist_ok=True)
    FIGURAS.mkdir(exist_ok=True)

    instancia = leer_instancia(DATOS / "CVRP_10.txt")
    demandas = instancia["demandas"]
    capacidad = instancia["capacidad"]
    matriz = calcular_matriz(instancia["coordenadas"])
    with open(RESULTADOS / "solucion_exacta.json", encoding="utf-8") as archivo:
        z_opt = json.load(archivo)["distancia_total"]

    titulo("ENTREGA 2: CONFIGURACIÓN")
    print(f"Instancia: {instancia['nombre']}   Z* = {z_opt!r} (resultados/solucion_exacta.json)")
    print(f"Semillas: {SEMILLAS.start}..{SEMILLAS.stop - 1}   éxito: |Z - Z*| < {TOLERANCIA_EXITO:g}")
    for algoritmo in ALGORITMOS:
        for configuracion, parametros in algoritmo["configuraciones"]:
            print(f"{algoritmo['siglas']:<3} {configuracion:<9}: {parametros}")

    # --- C1. Confirmación ---
    filas_confirmacion = []
    corridas_por_configuracion = {}
    for algoritmo in ALGORITMOS:
        for configuracion, parametros in algoritmo["configuraciones"]:
            corridas = correr(algoritmo, parametros, matriz, demandas, capacidad)
            filas_confirmacion.append(
                resumir_confirmacion(algoritmo, configuracion, corridas, z_opt))
            corridas_por_configuracion[algoritmo["clave"], configuracion] = corridas
    finales = {a["clave"]: corridas_por_configuracion[a["clave"], "FINAL"] for a in ALGORITMOS}
    guardar_confirmacion_csv(filas_confirmacion, RESULTADOS / "confirmacion_parametros.csv")
    guardar_confirmacion_tex(filas_confirmacion, RESULTADOS / "confirmacion_parametros.tex")

    titulo("C1. CONFIRMACIÓN DE PARÁMETROS: BASE, COMBINADA y FINAL (desv. muestrales)")
    imprimir_confirmacion(filas_confirmacion)
    print()
    imprimir_fallas("GA COMBINADA", ALGORITMOS[0], SEMILLAS,
                    corridas_por_configuracion["ga", "COMBINADA"], z_opt)

    # --- Verificación ampliada (las semillas 0..29 ya se corrieron en C1) ---
    filas_verificacion = []
    verificacion = {}
    for algoritmo in ALGORITMOS:
        semillas_nuevas = range(SEMILLAS.stop, SEMILLAS_VERIFICACION.stop)
        parametros = dict(algoritmo["configuraciones"])["FINAL"]
        corridas = finales[algoritmo["clave"]] + correr(
            algoritmo, parametros, matriz, demandas, capacidad, semillas_nuevas)
        verificacion[algoritmo["clave"]] = corridas
        filas_verificacion.append(resumir_verificacion(algoritmo, corridas, z_opt))
    guardar_verificacion_csv(filas_verificacion, RESULTADOS / "verificacion_100.csv")
    guardar_verificacion_tex(filas_verificacion, RESULTADOS / "verificacion_100.tex")

    titulo(f"VERIFICACIÓN AMPLIADA: CONFIGURACIÓN FINAL, SEMILLAS "
           f"{SEMILLAS_VERIFICACION.start}..{SEMILLAS_VERIFICACION.stop - 1}")
    imprimir_verificacion(filas_verificacion)
    print()
    for algoritmo in ALGORITMOS:
        imprimir_fallas(f"{algoritmo['siglas']} FINAL", algoritmo, SEMILLAS_VERIFICACION,
                        verificacion[algoritmo["clave"]], z_opt)

    # --- C2. Validación ---
    filas_validacion = [resumir_validacion(a, finales[a["clave"]], z_opt) for a in ALGORITMOS]
    guardar_validacion_tex(filas_validacion, RESULTADOS / "validacion_bio.tex")

    titulo("C2. VALIDACIÓN: CONFIGURACIÓN FINAL, 30 CORRIDAS")
    imprimir_validacion(filas_validacion)

    for algoritmo in ALGORITMOS:
        corridas = finales[algoritmo["clave"]]
        semilla = mejor_corrida(corridas)
        mejor = corridas[semilla]
        solucion = evaluar_solucion(mejor["rutas"], matriz, demandas, capacidad)
        solucion["semilla"] = semilla
        solucion["parametros"] = mejor["parametros"]
        solucion["convergencia"] = {
            clave: mejor[clave] for clave in mejor
            if clave not in ("rutas", "costo", "gran_ruta", "parametros")}
        if "gran_ruta" in mejor:
            solucion["gran_ruta"] = mejor["gran_ruta"]
        guardar_solucion_tex(solucion, RESULTADOS / f"solucion_{algoritmo['clave']}.tex")
        guardar_solucion_json(solucion, RESULTADOS / f"solucion_{algoritmo['clave']}.json")
        graficar_rutas(instancia, solucion, FIGURAS / f"rutas_{algoritmo['clave']}.pdf",
                       titulo=algoritmo["nombre"])

        titulo(f"RUTAS: {algoritmo['nombre'].upper()} (mejor corrida, semilla {semilla})")
        imprimir_rutas(solucion)
        print(f"{algoritmo['etiqueta_mejor'].capitalize()} del mejor: {mejor[algoritmo['clave_mejor']]}"
              f"   evaluaciones hasta el mejor: {mejor['evaluaciones_hasta_mejor']} "
              f"de {mejor['evaluaciones']}   tiempo: {mejor['tiempo'] * 1e3:.2f} ms")

    graficar_convergencia([
        {"titulo": a["nombre"], "eje_x": a["eje_x"], "inicio": a["inicio"],
         "historial_mejor": [c["historial_mejor"] for c in finales[a["clave"]]],
         "historial_promedio": [c["historial_promedio"] for c in finales[a["clave"]]]}
        for a in ALGORITMOS
    ], z_opt, FIGURAS / "convergencia.pdf")

    titulo("ARCHIVOS GENERADOS")
    for archivo in ("resultados/confirmacion_parametros.tex",
                    "resultados/confirmacion_parametros.csv", "resultados/verificacion_100.tex",
                    "resultados/verificacion_100.csv", "resultados/validacion_bio.tex",
                    "resultados/solucion_ga.tex", "resultados/solucion_ga.json",
                    "resultados/solucion_aco.tex", "resultados/solucion_aco.json",
                    "figuras/rutas_ga.pdf", "figuras/rutas_aco.pdf", "figuras/convergencia.pdf"):
        print(archivo)


if __name__ == "__main__":
    main()

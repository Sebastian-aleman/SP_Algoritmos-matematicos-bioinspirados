"""Calibración de hiperparámetros del GA y del ACO sobre CVRP_10, un parámetro a la vez."""

import csv
import json
import statistics
from pathlib import Path

from src.colonia_hormigas import colonia_hormigas
from src.genetico import algoritmo_genetico
from src.instancia import leer_instancia
from src.matriz_costos import calcular_matriz

RAIZ = Path(__file__).resolve().parent
DATOS = RAIZ / "datos"
RESULTADOS = RAIZ / "resultados"
SEMILLAS = range(30)
TOLERANCIA_EXITO = 1e-4

# Base congelada: los valores por defecto con los que se hizo la calibración. No se
# toman de PARAMETROS_GA / PARAMETROS_ACO para que la calibración siga siendo
# reproducible aunque esos valores por defecto cambien.
BASE_GA = {
    "tam_poblacion": 50,
    "generaciones": 200,
    "prob_cruza": 0.9,
    "prob_mutacion": 0.2,
    "tam_torneo": 2,
    "elitismo": 2,
    "paciencia": 100,
    "prob_busqueda_local": 0.0,
}

BASE_ACO = {
    "num_hormigas": None,
    "iteraciones": 100,
    "alfa": 5,
    "beta": 5,
    "rho": 0.75,
    "sigma": 6,
    "f": 2,
    "g": 2,
    "fraccion_candidatos": 0.25,
    "dos_opt": True,
    "paciencia": 50,
}

REJILLA_GA = {
    "tam_poblacion": [20, 30, 50, 100],
    "prob_cruza": [0.6, 0.8, 0.9, 1.0],
    "prob_mutacion": [0.05, 0.1, 0.2, 0.3, 0.5],
    "tam_torneo": [2, 3, 5],
    "elitismo": [0, 1, 2, 5],
}

REJILLA_ACO = {
    "alfa": [1, 2, 5],
    "beta": [1, 2, 5],
    "rho": [0.5, 0.75, 0.9],
    "sigma": [2, 4, 6, 10],
    "fraccion_candidatos": [0.25, 0.5, 1.0],
    "num_hormigas": [5, 10, 20],
    "dos_opt": [True, False],
}

COLUMNAS = ["parametro", "valor", "es_base", "exito_pct", "z_promedio", "z_desviacion",
            "z_maximo", "mejor_promedio", "mejor_desviacion", "mejor_maximo",
            "evaluaciones_hasta_mejor_promedio", "evaluaciones_hasta_mejor_desviacion",
            "tiempo_hasta_mejor_mediana", "tiempo_mediana"]


def titulo(texto):
    print(f"\n{'=' * 78}\n{texto}\n{'=' * 78}")


def resumir(corridas, z_opt, clave_mejor):
    """Métricas de las corridas de una configuración (desviaciones muestrales)."""
    costos = [c["costo"] for c in corridas]
    mejores = [c[clave_mejor] for c in corridas]
    evaluaciones = [c["evaluaciones_hasta_mejor"] for c in corridas]
    return {
        "exito_pct": sum(abs(z - z_opt) < TOLERANCIA_EXITO for z in costos) / len(costos) * 100,
        "z_promedio": statistics.mean(costos),
        "z_desviacion": statistics.stdev(costos),
        "z_maximo": max(costos),
        "mejor_promedio": statistics.mean(mejores),
        "mejor_desviacion": statistics.stdev(mejores),
        "mejor_maximo": max(mejores),
        "evaluaciones_hasta_mejor_promedio": statistics.mean(evaluaciones),
        "evaluaciones_hasta_mejor_desviacion": statistics.stdev(evaluaciones),
        "tiempo_hasta_mejor_mediana": statistics.median(c["tiempo_hasta_mejor"] for c in corridas),
        "tiempo_mediana": statistics.median(c["tiempo"] for c in corridas),
    }


def calibrar(algoritmo, base, rejilla, clave_mejor, matriz, demandas, capacidad, z_opt):
    """Corre cada valor de la rejilla con los demás parámetros en su valor base.

    La configuración base aparece en varios grupos; se corre una sola vez y se reutiliza.
    """
    numero_clientes = len(matriz) - 1
    resumenes = {}
    filas = []
    for parametro, valores in rejilla.items():
        valor_base = base[parametro]
        if parametro == "num_hormigas" and valor_base is None:
            valor_base = numero_clientes  # None equivale a una hormiga por cliente
        for valor in valores:
            configuracion = {**base, parametro: valor}
            clave = tuple(sorted(configuracion.items()))
            if clave not in resumenes:
                corridas = [algoritmo(matriz, demandas, capacidad, semilla=s, **configuracion)
                            for s in SEMILLAS]
                resumenes[clave] = resumir(corridas, z_opt, clave_mejor)
            filas.append({"parametro": parametro, "valor": valor,
                          "es_base": valor == valor_base, **resumenes[clave]})
    return filas


def ganadores(filas):
    """Por parámetro: mayor % de éxito, luego menos evaluaciones hasta el mejor,
    luego menor mediana de tiempo total."""
    grupos = {}
    for f in filas:
        grupos.setdefault(f["parametro"], []).append(f)
    return {parametro: min(grupo, key=lambda f: (-f["exito_pct"],
                                                 f["evaluaciones_hasta_mejor_promedio"],
                                                 f["tiempo_mediana"]))
            for parametro, grupo in grupos.items()}


def guardar_calibracion_csv(filas, ruta):
    with open(ruta, "w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=COLUMNAS)
        escritor.writeheader()
        escritor.writerows(filas)


def guardar_calibracion_tex(filas, ruta, etiqueta_mejor):
    """Tabla booktabs agrupada por parámetro; el valor base lleva asterisco."""
    lineas = [
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        rf"Parámetro & Valor & \% éxito & $Z$ prom. & {etiqueta_mejor} mejor (prom. $\pm$ desv.) & "
        r"Eval. hasta mejor (prom.) & Tiempo hasta mejor (ms, mediana) \\",
    ]
    anterior = None
    for f in filas:
        nombre = ""
        if f["parametro"] != anterior:
            lineas.append(r"\midrule")
            nombre = r"\texttt{" + f["parametro"].replace("_", r"\_") + "}"
            anterior = f["parametro"]
        valor = f"{f['valor']}*" if f["es_base"] else f"{f['valor']}"
        lineas.append(
            f"{nombre} & {valor} & {f['exito_pct']:.1f} & {f['z_promedio']:.2f} & "
            f"{f['mejor_promedio']:.2f} $\\pm$ {f['mejor_desviacion']:.2f} & "
            f"{f['evaluaciones_hasta_mejor_promedio']:.1f} & "
            f"{f['tiempo_hasta_mejor_mediana'] * 1e3:.2f} \\\\"
        )
    lineas += [
        r"\midrule",
        rf"\multicolumn{{7}}{{l}}{{* Valor base. {len(SEMILLAS)} semillas por configuración; "
        rf"éxito: $|Z - Z^*| < 10^{{-4}}$.}} \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    with open(ruta, "w", encoding="utf-8") as archivo:
        archivo.write("\n".join(lineas) + "\n")


def imprimir_calibracion(filas, etiqueta_mejor):
    print(f"{'parámetro':<20} {'valor':>6} {'% éxito':>7} {'Z prom.':>9} {'Z desv.':>7} "
          f"{'Z máx.':>8} {etiqueta_mejor + ' mejor (máx.)':>22} {'eval. hasta mejor':>17} "
          f"{'t mejor (ms)':>12} {'t total (ms)':>12}")
    anterior = None
    for f in filas:
        if anterior is not None and f["parametro"] != anterior:
            print()
        nombre = f["parametro"] if f["parametro"] != anterior else ""
        anterior = f["parametro"]
        valor = f"{f['valor']}*" if f["es_base"] else f"{f['valor']}"
        mejor = (f"{f['mejor_promedio']:.2f} ± {f['mejor_desviacion']:.2f} "
                 f"({f['mejor_maximo']})")
        evaluaciones = (f"{f['evaluaciones_hasta_mejor_promedio']:.1f} ± "
                        f"{f['evaluaciones_hasta_mejor_desviacion']:.1f}")
        print(f"{nombre:<20} {valor:>6} {f['exito_pct']:>7.1f} {f['z_promedio']:>9.4f} "
              f"{f['z_desviacion']:>7.4f} {f['z_maximo']:>8.2f} {mejor:>22} {evaluaciones:>17} "
              f"{f['tiempo_hasta_mejor_mediana'] * 1e3:>12.2f} {f['tiempo_mediana'] * 1e3:>12.2f}")


def imprimir_ganadores(filas):
    print("Criterio: 1) mayor % de éxito, 2) menor prom. de evaluaciones hasta el mejor, "
          "3) menor mediana de tiempo total.")
    for parametro, f in ganadores(filas).items():
        base = next(g["valor"] for g in filas if g["parametro"] == parametro and g["es_base"])
        cambio = "(= base)" if f["es_base"] else f"(base: {base})"
        print(f"{parametro:<20} -> {str(f['valor']):<6} {cambio:<12} éxito {f['exito_pct']:5.1f} %, "
              f"eval. hasta mejor {f['evaluaciones_hasta_mejor_promedio']:8.1f}, "
              f"t total {f['tiempo_mediana'] * 1e3:7.2f} ms")


def main():
    RESULTADOS.mkdir(exist_ok=True)

    instancia = leer_instancia(DATOS / "CVRP_10.txt")
    demandas = instancia["demandas"]
    capacidad = instancia["capacidad"]
    matriz = calcular_matriz(instancia["coordenadas"])
    with open(RESULTADOS / "solucion_exacta.json", encoding="utf-8") as archivo:
        z_opt = json.load(archivo)["distancia_total"]

    titulo("CALIBRACIÓN: CONFIGURACIÓN")
    print(f"Instancia: {instancia['nombre']}   Z* = {z_opt!r} (resultados/solucion_exacta.json)")
    print(f"Semillas por configuración: {SEMILLAS.start}..{SEMILLAS.stop - 1}   "
          f"éxito: |Z - Z*| < {TOLERANCIA_EXITO:g}")
    print(f"Base GA:  {BASE_GA}")
    print(f"Base ACO: {BASE_ACO}  (num_hormigas=None equivale a 10 = n)")

    filas_ga = calibrar(algoritmo_genetico, BASE_GA, REJILLA_GA, "generacion_mejor",
                        matriz, demandas, capacidad, z_opt)
    guardar_calibracion_csv(filas_ga, RESULTADOS / "calibracion_ga.csv")
    guardar_calibracion_tex(filas_ga, RESULTADOS / "calibracion_ga.tex", "Gen.")

    filas_aco = calibrar(colonia_hormigas, BASE_ACO, REJILLA_ACO, "iteracion_mejor",
                         matriz, demandas, capacidad, z_opt)
    guardar_calibracion_csv(filas_aco, RESULTADOS / "calibracion_aco.csv")
    guardar_calibracion_tex(filas_aco, RESULTADOS / "calibracion_aco.tex", "It.")

    titulo("CALIBRACIÓN GA (* = valor base; desviaciones muestrales)")
    imprimir_calibracion(filas_ga, "gen.")
    titulo("CALIBRACIÓN ACO (* = valor base; desviaciones muestrales)")
    imprimir_calibracion(filas_aco, "it.")

    titulo("GANADORES GA")
    imprimir_ganadores(filas_ga)
    titulo("GANADORES ACO")
    imprimir_ganadores(filas_aco)

    titulo("ARCHIVOS GENERADOS")
    for archivo in ("resultados/calibracion_ga.csv", "resultados/calibracion_ga.tex",
                    "resultados/calibracion_aco.csv", "resultados/calibracion_aco.tex"):
        print(archivo)


if __name__ == "__main__":
    main()

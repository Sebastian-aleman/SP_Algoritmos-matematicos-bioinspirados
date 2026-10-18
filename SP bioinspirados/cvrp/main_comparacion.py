"""Paso 5: comparación del modelo exacto, Clarke-Wright (+ búsqueda local), GA y ACO.

Todos los métodos se miden en la misma máquina y en la misma ejecución.
"""

import csv
import platform
import statistics
import subprocess
import sys
from pathlib import Path

import pulp

from main_entrega2 import es_exito, forma_canonica, mejor_corrida
from src.algoritmo import medir_tiempo, medir_tiempo_exacto, resolver_clarke_wright, resolver_cvrp
from src.colonia_hormigas import PARAMETROS_ACO, colonia_hormigas
from src.comparacion import construir_comparacion, desperdicio_global, guardar_comparacion_tex
from src.genetico import PARAMETROS_GA, algoritmo_genetico
from src.graficas import graficar_comparacion
from src.instancia import leer_instancia
from src.matriz_costos import calcular_matriz
from src.modelo_exacto import resolver_modelo_exacto
from src.reporte import evaluar_solucion

RAIZ = Path(__file__).resolve().parent
DATOS = RAIZ / "datos"
RESULTADOS = RAIZ / "resultados"
FIGURAS = RAIZ / "figuras"
REPETICIONES = 100
REPETICIONES_EXACTO = 5
SEMILLAS = range(30)

# Nombres con saltos de línea solo para la figura, para que las 5 etiquetas no se encimen
ETIQUETAS_FIGURA = {
    "Modelo exacto (PuLP/CBC)": "Modelo\nexacto\n(PuLP/CBC)",
    "Clarke-Wright": "Clarke-\nWright",
    "Clarke-Wright + búsqueda local": "Clarke-\nWright\n+ búsqueda\nlocal",
    "Algoritmo genético": "Algoritmo\ngenético",
    "Colonia de hormigas": "Colonia de\nhormigas",
}

COLUMNAS_EXTENDIDA = [
    "metodo", "z_mejor", "z_promedio", "exito_pct", "numero_rutas", "desperdicio_global_pct",
    "tiempo", "tiempo_hasta_mejor", "gap_mejor_pct", "gap_promedio_pct", "diferencia_tiempo_pct"]


def titulo(texto):
    print(f"\n{'=' * 78}\n{texto}\n{'=' * 78}")


def nombre_cpu():
    """Marca y modelo del procesador (Windows, macOS o Linux)."""
    try:
        if sys.platform == "win32":
            import winreg
            clave = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                   r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(clave, "ProcessorNameString")[0].strip()
        if sys.platform == "darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True, check=True).stdout.strip()
        with open("/proc/cpuinfo", encoding="utf-8") as archivo:
            for linea in archivo:
                if linea.startswith("model name"):
                    return linea.split(":", 1)[1].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or "desconocido"


def gap(z, z_opt):
    # round + 0.0 evita un "-0.00" por error de punto flotante, como en construir_comparacion
    return round((z - z_opt) / z_opt * 100, 9) + 0.0


def fila_extendida(nombre, solucion, tiempo, z_opt, t_opt, corridas=None):
    """Fila de la tabla extendida; sin corridas, el método es determinista."""
    z = solucion["distancia_total"]
    if corridas is None:
        z_promedio = z
        exito_pct = 100.0 if es_exito(z, z_opt) else 0.0
        tiempo_hasta_mejor = None
    else:
        costos = [c["costo"] for c in corridas]
        z_promedio = statistics.mean(costos)
        exito_pct = sum(es_exito(c, z_opt) for c in costos) / len(costos) * 100
        tiempo_hasta_mejor = statistics.median(c["tiempo_hasta_mejor"] for c in corridas)
    return {
        "metodo": nombre,
        "z_mejor": z,
        "z_promedio": z_promedio,
        "exito_pct": exito_pct,
        "numero_rutas": solucion["numero_rutas"],
        "desperdicio_global_pct": desperdicio_global(solucion),
        "tiempo": tiempo,
        "tiempo_hasta_mejor": tiempo_hasta_mejor,
        "gap_mejor_pct": gap(z, z_opt),
        "gap_promedio_pct": gap(z_promedio, z_opt),
        "diferencia_tiempo_pct": (tiempo - t_opt) / t_opt * 100,
    }


def guardar_extendida_csv(filas, ruta):
    with open(ruta, "w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=COLUMNAS_EXTENDIDA)
        escritor.writeheader()
        escritor.writerows(filas)


def guardar_extendida_tex(filas, ruta):
    lineas = [
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrrrrrr}",
        r"\toprule",
        r"Método & $Z$ mejor & $Z$ prom. & \% éxito & N.º rutas & Desperdicio global (\%) & "
        r"Tiempo total (s) & Tiempo hasta el mejor (s) & GAP mejor (\%) & GAP prom. (\%) & "
        r"Dif. tiempo (\%) \\",
        r"\midrule",
    ]
    for f in filas:
        hasta_mejor = "--" if f["tiempo_hasta_mejor"] is None else f"{f['tiempo_hasta_mejor']:.6f}"
        lineas.append(
            f"{f['metodo']} & {f['z_mejor']:.2f} & {f['z_promedio']:.2f} & {f['exito_pct']:.1f} & "
            f"{f['numero_rutas']} & {f['desperdicio_global_pct']:.2f} & {f['tiempo']:.6f} & "
            f"{hasta_mejor} & {f['gap_mejor_pct']:.2f} & {f['gap_promedio_pct']:.2f} & "
            f"{f['diferencia_tiempo_pct']:.4f} \\\\"
        )
    lineas += [
        r"\midrule",
        rf"\multicolumn{{11}}{{l}}{{Tiempos: mediana de {REPETICIONES_EXACTO} solves (exacto), "
        rf"{REPETICIONES} ejecuciones (Clarke-Wright) y {len(SEMILLAS)} semillas (GA y ACO, "
        rf"mejor solución de las {len(SEMILLAS)}). Éxito: $|Z - Z^*| < 10^{{-4}}$.}} \\",
        r"\bottomrule",
        r"\end{tabular}%",
        "}",
    ]
    with open(ruta, "w", encoding="utf-8") as archivo:
        archivo.write("\n".join(lineas) + "\n")


def imprimir_extendida(filas):
    print(f"{'método':<31} {'Z mejor':>8} {'Z prom.':>8} {'% éxito':>7} {'rutas':>5} "
          f"{'% desp.':>7} {'tiempo (s)':>11} {'t mejor (s)':>11} {'GAP mejor':>9} "
          f"{'GAP prom.':>9} {'dif. tiempo (%)':>16}")
    for f in filas:
        hasta_mejor = "--" if f["tiempo_hasta_mejor"] is None else f"{f['tiempo_hasta_mejor']:.6f}"
        print(f"{f['metodo']:<31} {f['z_mejor']:>8.2f} {f['z_promedio']:>8.2f} "
              f"{f['exito_pct']:>7.1f} {f['numero_rutas']:>5} {f['desperdicio_global_pct']:>7.2f} "
              f"{f['tiempo']:>11.6f} {hasta_mejor:>11} {f['gap_mejor_pct']:>9.2f} "
              f"{f['gap_promedio_pct']:>9.2f} {f['diferencia_tiempo_pct']:>16.4f}")


def comparar_rutas(nombre, rutas, rutas_opt):
    """Compara como conjunto de rutas, sin importar su orden ni su sentido."""
    propias, optimas = set(forma_canonica(rutas)), set(forma_canonica(rutas_opt))
    if propias == optimas:
        print(f"{nombre:<31} mismas rutas que el óptimo exacto")
        return

    def secuencias(conjunto):
        return ", ".join("0-" + "-".join(map(str, r)) + "-0" for r in sorted(conjunto))
    print(f"{nombre:<31} rutas DISTINTAS al óptimo exacto")
    print(f"{'':<31}   solo en el método: {secuencias(propias - optimas)}")
    print(f"{'':<31}   solo en el óptimo: {secuencias(optimas - propias)}")


def main():
    RESULTADOS.mkdir(exist_ok=True)
    FIGURAS.mkdir(exist_ok=True)

    instancia = leer_instancia(DATOS / "CVRP_10.txt")
    demandas = instancia["demandas"]
    capacidad = instancia["capacidad"]
    matriz = calcular_matriz(instancia["coordenadas"])

    titulo("EQUIPO")
    print(f"CPU:              {nombre_cpu()}")
    print(f"Sistema:          {platform.platform()}")
    print(f"Python:           {platform.python_version()} ({platform.python_implementation()})")
    print(f"PuLP:             {pulp.__version__}")

    # --- Modelo exacto (sin sobrescribir los archivos de la Entrega 1) ---
    rutas_exacta, modelo = resolver_modelo_exacto(
        matriz, demandas, capacidad, ruta_log=RESULTADOS / "cbc_comparacion.log")
    solucion_exacta = evaluar_solucion(rutas_exacta, matriz, demandas, capacidad)
    z_opt = solucion_exacta["distancia_total"]
    tiempo_exacto = medir_tiempo_exacto(matriz, demandas, capacidad, REPETICIONES_EXACTO)

    # --- Clarke-Wright y Clarke-Wright + búsqueda local ---
    solucion_cw = evaluar_solucion(resolver_clarke_wright(instancia), matriz, demandas, capacidad)
    rutas_alg, _ = resolver_cvrp(instancia)
    solucion_alg = evaluar_solucion(rutas_alg, matriz, demandas, capacidad)
    tiempo_cw = medir_tiempo(resolver_clarke_wright, instancia, REPETICIONES)
    tiempo_alg = medir_tiempo(resolver_cvrp, instancia, REPETICIONES)

    # --- GA y ACO con sus parámetros por defecto ---
    bioinspirados = {}
    for nombre, funcion in (("Algoritmo genético", algoritmo_genetico),
                            ("Colonia de hormigas", colonia_hormigas)):
        corridas = [funcion(matriz, demandas, capacidad, semilla=s) for s in SEMILLAS]
        semilla = mejor_corrida(corridas)
        bioinspirados[nombre] = {
            "corridas": corridas,
            "semilla": semilla,
            "solucion": evaluar_solucion(corridas[semilla]["rutas"], matriz, demandas, capacidad),
            "tiempo": statistics.median(c["tiempo"] for c in corridas),
        }

    titulo("MODELO EXACTO (PuLP / CBC)")
    print(f"Estatus:              {modelo['estatus']}")
    print(f"Z óptimo:             {z_opt:.4f}")
    print(f"Nodos explorados:     {modelo['nodos_explorados']}")
    print(f"Tiempo de solve:      {modelo['tiempo']:.4f} s (solve con log en "
          f"resultados/cbc_comparacion.log)")
    print(f"Tiempo (mediana de {REPETICIONES_EXACTO} solves): {tiempo_exacto:.4f} s")
    print(f"GA:  parámetros {PARAMETROS_GA}")
    print(f"ACO: parámetros {PARAMETROS_ACO}")
    for nombre, b in bioinspirados.items():
        print(f"{nombre}: mejor corrida = semilla {b['semilla']}")

    metodos = [
        ("Modelo exacto (PuLP/CBC)", solucion_exacta, tiempo_exacto),
        ("Clarke-Wright", solucion_cw, tiempo_cw),
        ("Clarke-Wright + búsqueda local", solucion_alg, tiempo_alg),
    ] + [(nombre, b["solucion"], b["tiempo"]) for nombre, b in bioinspirados.items()]

    # a) Tabla estándar
    filas = construir_comparacion(metodos)
    guardar_comparacion_tex(filas, RESULTADOS / "comparacion_entrega2.tex")
    titulo(f"a) COMPARACIÓN ESTÁNDAR (tiempo: mediana de {REPETICIONES_EXACTO} solves / "
           f"{REPETICIONES} ejecuciones / {len(SEMILLAS)} semillas)")
    print(f"{'método':<31} {'Z':>8} {'rutas':>5} {'% desp.':>8} {'tiempo (s)':>11} "
          f"{'GAP (%)':>8} {'dif. tiempo (%)':>16}")
    for f in filas:
        print(f"{f['metodo']:<31} {f['z']:>8.2f} {f['numero_rutas']:>5} "
              f"{f['desperdicio_global_pct']:>8.2f} {f['tiempo']:>11.6f} "
              f"{f['gap_costo_pct']:>8.2f} {f['diferencia_tiempo_pct']:>16.4f}")

    # b) Tabla extendida
    extendida = [fila_extendida(nombre, solucion, tiempo, z_opt, tiempo_exacto,
                                bioinspirados.get(nombre, {}).get("corridas"))
                 for nombre, solucion, tiempo in metodos]
    guardar_extendida_csv(extendida, RESULTADOS / "comparacion_entrega2_ext.csv")
    guardar_extendida_tex(extendida, RESULTADOS / "comparacion_entrega2_ext.tex")
    titulo("b) COMPARACIÓN EXTENDIDA")
    imprimir_extendida(extendida)

    # c) Figura
    graficar_comparacion([{**f, "metodo": ETIQUETAS_FIGURA[f["metodo"]]} for f in filas],
                         FIGURAS / "comparacion_entrega2.pdf")

    # d) Rutas contra el óptimo exacto
    titulo("d) ¿MISMAS RUTAS QUE EL ÓPTIMO EXACTO? (sin importar orden ni sentido)")
    print("Óptimo exacto: " + ", ".join("-".join(map(str, r["secuencia"]))
                                         for r in solucion_exacta["rutas"]))
    for nombre, solucion, _ in metodos[1:]:
        comparar_rutas(nombre, [r["secuencia"][1:-1] for r in solucion["rutas"]], rutas_exacta)

    titulo("ARCHIVOS GENERADOS")
    for archivo in ("resultados/cbc_comparacion.log", "resultados/comparacion_entrega2.tex",
                    "resultados/comparacion_entrega2_ext.tex",
                    "resultados/comparacion_entrega2_ext.csv",
                    "figuras/comparacion_entrega2.pdf"):
        print(archivo)


if __name__ == "__main__":
    main()

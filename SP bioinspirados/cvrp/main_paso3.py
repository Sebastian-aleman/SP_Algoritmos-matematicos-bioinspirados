"""Paso 3: modelo exacto (PuLP/CBC), Clarke-Wright + búsqueda local y comparación."""

from pathlib import Path

from src.algoritmo import medir_tiempo, resolver_clarke_wright, resolver_cvrp
from src.comparacion import construir_comparacion, guardar_comparacion_tex
from src.graficas import graficar_comparacion, graficar_rutas
from src.instancia import leer_instancia
from src.matriz_costos import calcular_matriz
from src.modelo_exacto import resolver_modelo_exacto
from src.reporte import evaluar_solucion, guardar_solucion_json, guardar_solucion_tex

RAIZ = Path(__file__).resolve().parent
DATOS = RAIZ / "datos"
RESULTADOS = RAIZ / "resultados"
FIGURAS = RAIZ / "figuras"
REPETICIONES = 100


def titulo(texto):
    print(f"\n{'=' * 78}\n{texto}\n{'=' * 78}")


def imprimir_rutas(solucion):
    print(f"{'ruta':>4}  {'secuencia':<16} {'carga':>5} {'distancia':>10} {'% desperdicio':>14}")
    for r in solucion["rutas"]:
        print(f"{r['ruta']:>4}  {'-'.join(map(str, r['secuencia'])):<16} {r['carga']:>5} "
              f"{r['distancia']:>10.2f} {r['desperdicio_pct']:>14.2f}")
    print(f"Z = {solucion['distancia_total']:.4f}   rutas = {solucion['numero_rutas']}   "
          f"demanda total = {solucion['demanda_total']}")


def main():
    RESULTADOS.mkdir(exist_ok=True)
    FIGURAS.mkdir(exist_ok=True)

    instancia = leer_instancia(DATOS / "CVRP_10.txt")
    demandas = instancia["demandas"]
    capacidad = instancia["capacidad"]
    matriz = calcular_matriz(instancia["coordenadas"])

    # --- Modelo exacto ---
    rutas_exacta, modelo = resolver_modelo_exacto(
        matriz, demandas, capacidad, ruta_log=RESULTADOS / "cbc_modelo_exacto.log")
    solucion_exacta = evaluar_solucion(rutas_exacta, matriz, demandas, capacidad)
    solucion_exacta["modelo"] = modelo
    guardar_solucion_tex(solucion_exacta, RESULTADOS / "solucion_exacta.tex")
    guardar_solucion_json(solucion_exacta, RESULTADOS / "solucion_exacta.json")

    titulo("MODELO EXACTO (PuLP / CBC)")
    print(f"Variables:            {modelo['numero_variables']} "
          f"({modelo['variables_binarias']} binarias x, {modelo['variables_continuas']} continuas u)")
    print(f"Restricciones:        {modelo['numero_restricciones']} "
          f"(más las cotas q_i <= u_i <= Q de las {modelo['variables_continuas']} variables u)")
    print(f"Estatus:              {modelo['estatus']}")
    print(f"Z óptimo:             {modelo['z']:.4f}")
    print(f"Cota inferior final:  {modelo['cota_inferior']:.4f}")
    if modelo["relajacion_lineal"] is not None:
        print(f"Relajación lineal:    {modelo['relajacion_lineal']:.4f}")
    print(f"Nodos explorados:     {modelo['nodos_explorados']}")
    print(f"Tiempo de solve:      {modelo['tiempo']:.4f} s")

    titulo("RUTAS: MODELO EXACTO")
    imprimir_rutas(solucion_exacta)

    # --- Clarke-Wright y Clarke-Wright + búsqueda local ---
    rutas_cw = resolver_clarke_wright(instancia)
    rutas_alg, movimientos = resolver_cvrp(instancia)
    solucion_cw = evaluar_solucion(rutas_cw, matriz, demandas, capacidad)
    solucion_alg = evaluar_solucion(rutas_alg, matriz, demandas, capacidad)

    tiempo_cw = medir_tiempo(resolver_clarke_wright, instancia, REPETICIONES)
    tiempo_alg = medir_tiempo(resolver_cvrp, instancia, REPETICIONES)

    solucion_alg["movimientos"] = movimientos
    solucion_alg["tiempos"] = {"repeticiones": REPETICIONES,
                               "mediana_clarke_wright": tiempo_cw,
                               "mediana_clarke_wright_busqueda_local": tiempo_alg}
    guardar_solucion_tex(solucion_alg, RESULTADOS / "solucion_algoritmo.tex")
    guardar_solucion_json(solucion_alg, RESULTADOS / "solucion_algoritmo.json")

    titulo("RUTAS: CLARKE-WRIGHT")
    imprimir_rutas(solucion_cw)

    titulo("BÚSQUEDA LOCAL: MOVIMIENTOS APLICADOS")
    print(f"{'#':>3}  {'tipo':<9} {'nodos':<9} {'ahorro':>8} {'Z tras mover':>13}  detalle")
    z = solucion_cw["distancia_total"]
    print(f"{0:>3}  {'(inicio)':<9} {'':<9} {'':>8} {z:>13.4f}  solución de Clarke-Wright")
    for numero, m in enumerate(movimientos, start=1):
        z -= m["ahorro"]
        nodos = ",".join(map(str, m["nodos"]))
        print(f"{numero:>3}  {m['tipo']:<9} {nodos:<9} {m['ahorro']:>8.4f} {z:>13.4f}  {m['detalle']}")
    print(f"Movimientos: {len(movimientos)}   ahorro total = "
          f"{sum(m['ahorro'] for m in movimientos):.4f}")

    titulo("RUTAS: CLARKE-WRIGHT + BÚSQUEDA LOCAL")
    imprimir_rutas(solucion_alg)

    # --- Comparación ---
    filas = construir_comparacion([
        ("Modelo exacto (PuLP/CBC)", solucion_exacta, modelo["tiempo"]),
        ("Clarke-Wright", solucion_cw, tiempo_cw),
        ("Clarke-Wright + búsqueda local", solucion_alg, tiempo_alg),
    ])
    guardar_comparacion_tex(filas, RESULTADOS / "comparacion.tex")

    titulo(f"COMPARACIÓN (tiempo heurísticos = mediana de {REPETICIONES} ejecuciones)")
    print(f"{'método':<31} {'Z':>8} {'rutas':>5} {'% desp.':>8} {'tiempo (s)':>11} "
          f"{'GAP (%)':>8} {'dif. tiempo (%)':>16}")
    for f in filas:
        print(f"{f['metodo']:<31} {f['z']:>8.2f} {f['numero_rutas']:>5} "
              f"{f['desperdicio_global_pct']:>8.2f} {f['tiempo']:>11.6f} "
              f"{f['gap_costo_pct']:>8.2f} {f['diferencia_tiempo_pct']:>16.4f}")

    graficar_rutas(instancia, solucion_exacta, FIGURAS / "rutas_exacta.pdf",
                   titulo="Modelo exacto")
    graficar_rutas(instancia, solucion_alg, FIGURAS / "rutas_algoritmo.pdf",
                   titulo="Clarke-Wright + búsqueda local")
    graficar_comparacion(filas, FIGURAS / "comparacion.pdf")

    titulo("ARCHIVOS GENERADOS")
    for archivo in ("resultados/solucion_exacta.tex", "resultados/solucion_exacta.json",
                    "resultados/cbc_modelo_exacto.log", "resultados/solucion_algoritmo.tex",
                    "resultados/solucion_algoritmo.json", "resultados/comparacion.tex",
                    "figuras/rutas_exacta.pdf", "figuras/rutas_algoritmo.pdf",
                    "figuras/comparacion.pdf"):
        print(archivo)


if __name__ == "__main__":
    main()

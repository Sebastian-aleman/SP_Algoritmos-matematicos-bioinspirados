"""Paso 1: instancia, matriz de costos, Clarke-Wright y reporte de la solución."""

from pathlib import Path

from src.clarke_wright import (calcular_ahorros, clarke_wright,
                               guardar_ahorros_tex, guardar_iteraciones_tex)
from src.graficas import graficar_nodos, graficar_rutas
from src.instancia import leer_instancia
from src.matriz_costos import calcular_matriz, guardar_matriz_csv, guardar_matriz_tex
from src.reporte import evaluar_solucion, guardar_solucion_json, guardar_solucion_tex

RAIZ = Path(__file__).resolve().parent
DATOS = RAIZ / "datos"
RESULTADOS = RAIZ / "resultados"
FIGURAS = RAIZ / "figuras"


def titulo(texto):
    print(f"\n{'=' * 78}\n{texto}\n{'=' * 78}")


def main():
    RESULTADOS.mkdir(exist_ok=True)
    FIGURAS.mkdir(exist_ok=True)

    instancia = leer_instancia(DATOS / "CVRP_10.txt")
    coordenadas = instancia["coordenadas"]
    demandas = instancia["demandas"]
    capacidad = instancia["capacidad"]
    n = instancia["dimension"]

    titulo(f"INSTANCIA {instancia['nombre']}  (Q = {capacidad})")
    print(f"{'nodo':>4} {'x':>7} {'y':>7} {'demanda':>8}  tipo")
    for i, ((x, y), q) in enumerate(zip(coordenadas, demandas)):
        print(f"{i:>4} {x:>7.2f} {y:>7.2f} {q:>8}  {'depósito' if i == 0 else 'cliente'}")

    matriz = calcular_matriz(coordenadas)
    guardar_matriz_csv(matriz, RESULTADOS / "matriz_costos.csv")
    guardar_matriz_tex(matriz, RESULTADOS / "matriz_costos.tex")

    titulo("MATRIZ DE COSTOS (distancia euclidiana, mostrada con 2 decimales)")
    print("     " + "".join(f"{j:>7}" for j in range(n)))
    for i in range(n):
        print(f"{i:>4} " + "".join(f"{c:>7.2f}" for c in matriz[i]))

    ahorros = calcular_ahorros(matriz)
    guardar_ahorros_tex(matriz, RESULTADOS / "ahorros.tex")

    titulo("TOP 15 AHORROS  s_ij = c_0i + c_0j - c_ij")
    print(f"{'#':>3} {'(i,j)':>8} {'c_0i':>8} {'c_0j':>8} {'c_ij':>8} {'s_ij':>8}")
    for posicion, (i, j, s) in enumerate(ahorros[:15], start=1):
        print(f"{posicion:>3} {f'({i},{j})':>8} {matriz[0][i]:>8.2f} "
              f"{matriz[0][j]:>8.2f} {matriz[i][j]:>8.2f} {s:>8.2f}")

    rutas, registro = clarke_wright(matriz, demandas, capacidad)
    guardar_iteraciones_tex(registro, RESULTADOS / "iteraciones_cw.tex")

    titulo("CLARKE-WRIGHT PARALELO: LOG DE FUSIONES")
    print(f"{'iter':>4} {'(i,j)':>8} {'s_ij':>8}  {'fusión':<6}  {'motivo':<30} ruta resultante")
    for e in registro:
        par = f"({e['i']},{e['j']})"
        resultante = "-".join(map(str, [0] + e["ruta"] + [0])) if e["fusionado"] else ""
        print(f"{e['iteracion']:>4} {par:>8} {e['ahorro']:>8.2f}  "
              f"{'SÍ' if e['fusionado'] else 'no':<6}  {e['motivo']:<30} {resultante}")

    solucion = evaluar_solucion(rutas, matriz, demandas, capacidad)
    guardar_solucion_tex(solucion, RESULTADOS / "solucion_manual.tex")
    guardar_solucion_json(solucion, RESULTADOS / "solucion_manual.json")

    titulo("SOLUCIÓN")
    print(f"{'ruta':>4}  {'secuencia':<16} {'carga':>5} {'distancia':>10} {'% desperdicio':>14}")
    for r in solucion["rutas"]:
        print(f"{r['ruta']:>4}  {'-'.join(map(str, r['secuencia'])):<16} {r['carga']:>5} "
              f"{r['distancia']:>10.2f} {r['desperdicio_pct']:>14.2f}")
    print(f"\nDistancia total (FO): {solucion['distancia_total']:.4f}")
    print(f"Número de rutas:      {solucion['numero_rutas']}")
    print(f"Demanda total:        {solucion['demanda_total']}")
    print(f"Vehículos mínimos:    ceil({solucion['demanda_total']}/{capacidad}) = "
          f"{solucion['vehiculos_minimos']}")

    graficar_nodos(instancia, FIGURAS / "nodos.pdf")
    graficar_rutas(instancia, solucion, FIGURAS / "rutas_manual.pdf",
                   titulo="Clarke-Wright")

    titulo("ARCHIVOS GENERADOS")
    for carpeta in (RESULTADOS, FIGURAS):
        for archivo in sorted(carpeta.iterdir()):
            print(archivo.relative_to(RAIZ))


if __name__ == "__main__":
    main()

"""Cota inferior TSP y partición (split) de la gran ruta."""

from pathlib import Path

from src.graficas import graficar_tsp
from src.instancia import leer_instancia
from src.matriz_costos import calcular_matriz
from src.tsp import held_karp, split

RAIZ = Path(__file__).resolve().parent
FIGURAS = RAIZ / "figuras"


def main():
    FIGURAS.mkdir(exist_ok=True)
    instancia = leer_instancia(RAIZ / "datos" / "CVRP_10.txt")
    matriz = calcular_matriz(instancia["coordenadas"])
    z_tsp, recorrido = held_karp(matriz)
    z_split, rutas = split(recorrido, matriz, instancia["demandas"], instancia["capacidad"])
    print(f"TSP óptimo: {z_tsp:.4f}  recorrido: 0-{'-'.join(map(str, recorrido))}-0")
    print(f"Split de la gran ruta TSP: Z = {z_split:.4f}  rutas = {rutas}")
    graficar_tsp(instancia, recorrido, z_tsp, FIGURAS / "rutas_tsp.pdf")


if __name__ == "__main__":
    main()
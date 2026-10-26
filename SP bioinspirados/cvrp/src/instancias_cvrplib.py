"""Lectura de instancias de CVRPLIB (series CMT y X) y utilidades comunes de la Etapa 3.

Diferencias con instancia.py (Entregas 1 y 2):
- Admite encabezados con tabuladores y fin de línea \\r\\n (formato de la serie X).
- Lee DISTANCE (duración máxima de ruta) y SERVICE_TIME, presentes en CMT6-10, 13 y 14.
- Toma la BKS de COMMENT en las CMT y de BKS_X en las X (CVRPLIB, consultado en 2026).
- Convención de distancias de CVRPLIB: las X usan la distancia euclidiana redondeada al
  entero más cercano; las CMT, la distancia euclidiana sin redondear.
"""

import math
from pathlib import Path

# Mejores soluciones conocidas de la serie X (CVRPLIB). Las CMT la traen en COMMENT.
BKS_X = {
    "X-n280-k17": 33503, "X-n303-k21": 21736, "X-n308-k13": 25859,
    "X-n327-k20": 27532, "X-n336-k84": 139111, "X-n401-k29": 66154,
    "X-n411-k19": 19712, "X-n429-k61": 65449, "X-n449-k29": 55233,
    "X-n459-k26": 24139,
}

INSTANCIAS_CMT = [f"CMT{k}" for k in range(1, 15)]
INSTANCIAS_X = list(BKS_X)
INSTANCIAS = INSTANCIAS_CMT + INSTANCIAS_X


def _numero(texto):
    try:
        return float(texto)
    except ValueError:
        return None


def leer_vrp(ruta):
    """Lee una instancia .vrp de CVRPLIB y devuelve un diccionario listo para los algoritmos.

    Claves: nombre, n (clientes), capacidad, demandas, coordenadas, matriz, limite_ruta
    (None si no hay), tiempo_servicio, bks, redondeo (bool), archivo.
    """
    ruta = Path(ruta)
    encabezado, coordenadas, demandas, depositos = {}, {}, {}, []
    seccion = None
    with open(ruta, encoding="utf-8") as archivo:
        for linea in archivo:
            linea = linea.strip()
            if not linea:
                continue
            if linea == "EOF":
                break
            if linea.endswith("_SECTION"):
                seccion = linea
                continue
            if seccion is None:
                clave, _, valor = linea.partition(":")
                encabezado[clave.strip()] = valor.strip().strip('"').strip()
            elif seccion == "NODE_COORD_SECTION":
                nodo, x, y = linea.split()
                coordenadas[int(nodo) - 1] = (float(x), float(y))
            elif seccion == "DEMAND_SECTION":
                nodo, demanda = linea.split()
                demandas[int(nodo) - 1] = int(float(demanda))
            elif seccion == "DEPOT_SECTION":
                if int(linea) != -1:
                    depositos.append(int(linea) - 1)

    dimension = int(encabezado["DIMENSION"])
    if encabezado.get("EDGE_WEIGHT_TYPE") != "EUC_2D":
        raise ValueError("Solo se admite EDGE_WEIGHT_TYPE : EUC_2D")
    if depositos != [0]:
        raise ValueError("Se espera un único depósito en el nodo 1 del archivo")
    if len(coordenadas) != dimension or len(demandas) != dimension:
        raise ValueError("El número de nodos no coincide con DIMENSION")

    nombre = encabezado.get("NAME", ruta.stem).strip()
    redondeo = nombre.startswith("X-")
    bks = BKS_X.get(nombre, _numero(encabezado.get("COMMENT", "")))
    limite = _numero(encabezado.get("DISTANCE", "")) if "DISTANCE" in encabezado else None
    servicio = _numero(encabezado.get("SERVICE_TIME", "0")) or 0.0
    coords = [coordenadas[k] for k in range(dimension)]
    return {
        "nombre": nombre,
        "n": dimension - 1,
        "capacidad": int(float(encabezado["CAPACITY"])),
        "demandas": [demandas[k] for k in range(dimension)],
        "coordenadas": coords,
        "matriz": calcular_matriz_vrp(coords, redondeo),
        "limite_ruta": limite,
        "tiempo_servicio": servicio,
        "bks": bks,
        "redondeo": redondeo,
        "archivo": str(ruta),
    }


def calcular_matriz_vrp(coordenadas, redondeo=False):
    """Matriz euclidiana; con redondeo usa nint (convención TSPLIB/CVRPLIB de la serie X)."""
    n = len(coordenadas)
    matriz = [[0.0] * n for _ in range(n)]
    for i in range(n):
        xi, yi = coordenadas[i]
        fila = matriz[i]
        for j in range(i + 1, n):
            xj, yj = coordenadas[j]
            d = math.hypot(xi - xj, yi - yj)
            if redondeo:
                d = float(int(d + 0.5))
            fila[j] = d
            matriz[j][i] = d
    return matriz


def subinstancia(instancia, k):
    """Depósito y los primeros k clientes de una instancia (misma capacidad y reglas)."""
    if not 1 <= k <= instancia["n"]:
        raise ValueError("k fuera de rango")
    nodos = range(k + 1)
    coords = [instancia["coordenadas"][i] for i in nodos]
    return {
        **instancia,
        "nombre": f"{instancia['nombre']}-k{k}",
        "n": k,
        "demandas": [instancia["demandas"][i] for i in nodos],
        "coordenadas": coords,
        "matriz": [[instancia["matriz"][i][j] for j in nodos] for i in nodos],
        "bks": None,
    }


def cargar(nombre, carpeta):
    """Carga una instancia por nombre; 'CMT1-k20' crea la subinstancia de CMT1 con 20 clientes."""
    base, _, sufijo = nombre.partition("-k") if nombre.startswith("CMT") else (nombre, "", "")
    instancia = leer_vrp(Path(carpeta) / f"{base}.vrp")
    return subinstancia(instancia, int(sufijo)) if sufijo else instancia


# --- Evaluación y validación de soluciones ---

def distancia_ruta(ruta, matriz):
    anterior, total = 0, 0.0
    for cliente in ruta:
        total += matriz[anterior][cliente]
        anterior = cliente
    return total + matriz[anterior][0]


def costo_solucion(rutas, matriz):
    return sum(distancia_ruta(r, matriz) for r in rutas)


def duracion_ruta(ruta, instancia):
    return distancia_ruta(ruta, instancia["matriz"]) + instancia["tiempo_servicio"] * len(ruta)


def validar(rutas, instancia, tolerancia=1e-6):
    """Revisa que cada cliente se visite una vez y que se respeten capacidad y duración.

    Devuelve (es_valida, mensaje, costo recalculado).
    """
    n = instancia["n"]
    visitados = [c for r in rutas for c in r]
    if sorted(visitados) != list(range(1, n + 1)):
        return False, "clientes faltantes o repetidos", math.inf
    for r in rutas:
        if not r:
            return False, "ruta vacía", math.inf
        if sum(instancia["demandas"][c] for c in r) > instancia["capacidad"]:
            return False, "capacidad excedida", math.inf
        if (instancia["limite_ruta"] is not None
                and duracion_ruta(r, instancia) > instancia["limite_ruta"] + tolerancia):
            return False, "duración máxima excedida", math.inf
    return True, "ok", costo_solucion(rutas, instancia["matriz"])


def vecinos_cercanos(matriz):
    """Para cada nodo, los clientes ordenados por distancia (sin incluirse a sí mismo)."""
    n = len(matriz)
    clientes = range(1, n)
    return [sorted((j for j in clientes if j != i), key=lambda j: (matriz[i][j], j))
            for i in range(n)]

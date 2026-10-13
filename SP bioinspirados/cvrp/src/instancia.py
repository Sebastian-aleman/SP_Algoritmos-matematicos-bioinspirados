"""Lectura de instancias CVRP en formato TSPLIB."""


def leer_instancia(ruta):
    """Lee una instancia CVRP (TSPLIB, EUC_2D) y la re-indexa desde 0.

    El nodo k del archivo pasa a ser el nodo k-1, de modo que el depósito
    queda en 0 y los clientes en 1..n.

    Devuelve un diccionario con: nombre, tipo, dimension, capacidad,
    coordenadas (lista de tuplas (x, y)), demandas (lista) y deposito (0).
    """
    encabezado = {}
    coordenadas = {}
    demandas = {}
    depositos = []
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
                encabezado[clave.strip()] = valor.strip()
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

    return {
        "nombre": encabezado.get("NAME", ""),
        "tipo": encabezado.get("TYPE", ""),
        "dimension": dimension,
        "capacidad": int(encabezado["CAPACITY"]),
        "coordenadas": [coordenadas[k] for k in range(dimension)],
        "demandas": [demandas[k] for k in range(dimension)],
        "deposito": 0,
    }

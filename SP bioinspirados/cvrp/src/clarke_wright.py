"""Heurística de ahorros de Clarke y Wright, versión paralela."""


def calcular_ahorros(matriz):
    """Calcula s_ij = c_0i + c_0j - c_ij para i<j, ordenados de mayor a menor.

    Los empates se resuelven por orden lexicográfico del par (i, j).
    """
    n = len(matriz)
    ahorros = [
        (i, j, matriz[0][i] + matriz[0][j] - matriz[i][j])
        for i in range(1, n)
        for j in range(i + 1, n)
    ]
    ahorros.sort(key=lambda a: (-a[2], a[0], a[1]))
    return ahorros


def clarke_wright(matriz, demandas, capacidad):
    """Construye rutas fusionando en orden decreciente de ahorro.

    Devuelve (rutas, registro). Cada ruta es la lista de clientes en orden de
    visita, sin el depósito. El registro tiene una entrada por ahorro evaluado:
    iteracion, i, j, ahorro, fusionado, motivo y ruta (resultante, si se fusionó).
    """
    n = len(matriz)
    rutas = {i: [i] for i in range(1, n)}  # id de ruta -> clientes
    ruta_de = {i: i for i in range(1, n)}  # cliente -> id de ruta
    carga = {i: demandas[i] for i in range(1, n)}
    registro = []

    for iteracion, (i, j, ahorro) in enumerate(calcular_ahorros(matriz), start=1):
        ri, rj = ruta_de[i], ruta_de[j]
        ruta_i, ruta_j = rutas[ri], rutas[rj]
        entrada = {"iteracion": iteracion, "i": i, "j": j, "ahorro": ahorro,
                   "fusionado": False, "motivo": "", "ruta": None}

        if ri == rj:
            entrada["motivo"] = "misma ruta"
        elif i not in (ruta_i[0], ruta_i[-1]) or j not in (ruta_j[0], ruta_j[-1]):
            interiores = [k for k, r in ((i, ruta_i), (j, ruta_j)) if k not in (r[0], r[-1])]
            entrada["motivo"] = "nodo interior (" + ", ".join(map(str, interiores)) + ")"
        elif carga[ri] + carga[rj] > capacidad:
            entrada["motivo"] = f"capacidad excedida ({carga[ri]}+{carga[rj]}>{capacidad})"
        else:
            # Orientar para que i quede al final de su ruta y j al inicio de la suya
            if ruta_i[-1] != i:
                ruta_i.reverse()
            if ruta_j[0] != j:
                ruta_j.reverse()
            ruta_i.extend(ruta_j)
            carga[ri] += carga[rj]
            for k in ruta_j:
                ruta_de[k] = ri
            del rutas[rj], carga[rj]
            entrada["fusionado"] = True
            entrada["motivo"] = f"carga {carga[ri]}/{capacidad}"
            entrada["ruta"] = list(ruta_i)

        registro.append(entrada)

    return list(rutas.values()), registro


def _secuencia(ruta):
    return "-".join(str(k) for k in [0] + ruta + [0])


def guardar_ahorros_tex(matriz, ruta, cantidad=15):
    """Guarda la tabla de los `cantidad` mayores ahorros."""
    lineas = [
        r"\begin{tabular}{ccrrrr}",
        r"\toprule",
        r"\# & $(i,j)$ & $c_{0i}$ & $c_{0j}$ & $c_{ij}$ & $s_{ij}$ \\",
        r"\midrule",
    ]
    for posicion, (i, j, ahorro) in enumerate(calcular_ahorros(matriz)[:cantidad], start=1):
        lineas.append(
            f"{posicion} & $({i},{j})$ & {matriz[0][i]:.2f} & {matriz[0][j]:.2f} & "
            f"{matriz[i][j]:.2f} & {ahorro:.2f} \\\\"
        )
    lineas += [r"\bottomrule", r"\end{tabular}"]
    with open(ruta, "w", encoding="utf-8") as archivo:
        archivo.write("\n".join(lineas) + "\n")


def guardar_iteraciones_tex(registro, ruta):
    """Guarda la tabla del proceso de fusión (una fila por ahorro evaluado)."""
    lineas = [
        r"\begin{tabular}{ccrcll}",
        r"\toprule",
        r"Iter. & $(i,j)$ & $s_{ij}$ & ¿Fusión? & Motivo & Ruta resultante \\",
        r"\midrule",
    ]
    for e in registro:
        decision = "Sí" if e["fusionado"] else "No"
        motivo = e["motivo"].replace(">", "$>$")
        resultante = _secuencia(e["ruta"]) if e["fusionado"] else "--"
        lineas.append(
            f"{e['iteracion']} & $({e['i']},{e['j']})$ & {e['ahorro']:.2f} & "
            f"{decision} & {motivo} & {resultante} \\\\"
        )
    lineas += [r"\bottomrule", r"\end{tabular}"]
    with open(ruta, "w", encoding="utf-8") as archivo:
        archivo.write("\n".join(lineas) + "\n")

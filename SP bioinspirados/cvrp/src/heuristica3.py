"""Algoritmo propuesto (Entrega 1) para instancias grandes: Clarke-Wright + búsqueda local.

Es el mismo algoritmo de clarke_wright.py y busqueda_local.py, con dos cambios para
escalar a cientos de clientes y a las instancias CMT con duración máxima de ruta:
- Ambas fases respetan la duración máxima (distancia + tiempo de servicio <= DISTANCE).
- La búsqueda local es granular (Toth y Vigo, 2003): relocate y swap solo se evalúan
  entre un cliente y sus K vecinos más cercanos, y tras cada movimiento se sigue
  recorriendo en lugar de reiniciar. Con K >= n - 1 los vecindarios son los mismos de
  la Entrega 1.
"""

import time

from src.instancias_cvrplib import distancia_ruta, vecinos_cercanos

TOLERANCIA = 1e-9
VECINOS_GRANULAR = 30


def _cabe(longitud, clientes, instancia):
    limite = instancia["limite_ruta"]
    return limite is None or longitud + instancia["tiempo_servicio"] * clientes <= limite + 1e-9


def clarke_wright3(instancia):
    """Clarke-Wright paralelo con capacidad y duración máxima. Devuelve la lista de rutas."""
    matriz, demandas, capacidad = instancia["matriz"], instancia["demandas"], instancia["capacidad"]
    n = instancia["n"]
    ahorros = [(matriz[0][i] + matriz[0][j] - matriz[i][j], i, j)
               for i in range(1, n + 1) for j in range(i + 1, n + 1)]
    ahorros.sort(key=lambda a: (-a[0], a[1], a[2]))

    rutas = {i: [i] for i in range(1, n + 1)}
    ruta_de = {i: i for i in range(1, n + 1)}
    carga = {i: demandas[i] for i in range(1, n + 1)}
    longitud = {i: 2 * matriz[0][i] for i in range(1, n + 1)}

    for ahorro, i, j in ahorros:
        if ahorro <= 0:
            break
        ri, rj = ruta_de[i], ruta_de[j]
        if ri == rj:
            continue
        ruta_i, ruta_j = rutas[ri], rutas[rj]
        if i not in (ruta_i[0], ruta_i[-1]) or j not in (ruta_j[0], ruta_j[-1]):
            continue
        if carga[ri] + carga[rj] > capacidad:
            continue
        nueva_longitud = longitud[ri] + longitud[rj] - ahorro
        if not _cabe(nueva_longitud, len(ruta_i) + len(ruta_j), instancia):
            continue
        if ruta_i[-1] != i:
            ruta_i.reverse()
        if ruta_j[0] != j:
            ruta_j.reverse()
        ruta_i.extend(ruta_j)
        carga[ri] += carga[rj]
        longitud[ri] = nueva_longitud
        for k in ruta_j:
            ruta_de[k] = ri
        del rutas[rj], carga[rj], longitud[rj]
    return list(rutas.values())


def dos_opt_ruta(ruta, matriz):
    """2-opt con mejor mejora sobre una ruta, hasta un óptimo local."""
    sec = [0] + list(ruta) + [0]
    while True:
        mejor_delta, mejor_par = -TOLERANCIA, None
        for a in range(1, len(sec) - 2):
            sa_1, sa = sec[a - 1], sec[a]
            for b in range(a + 1, len(sec) - 1):
                delta = (matriz[sa_1][sec[b]] + matriz[sa][sec[b + 1]]
                         - matriz[sa_1][sa] - matriz[sec[b]][sec[b + 1]])
                if delta < mejor_delta:
                    mejor_delta, mejor_par = delta, (a, b)
        if mejor_par is None:
            return sec[1:-1]
        a, b = mejor_par
        sec[a:b + 1] = reversed(sec[a:b + 1])


def busqueda_local3(rutas, instancia, vecinos=None, k_vecinos=VECINOS_GRANULAR):
    """2-opt intra-ruta, relocate y swap inter-ruta (granulares), respetando capacidad y duración.

    Devuelve (rutas, número de movimientos aplicados).
    """
    matriz, demandas, capacidad = instancia["matriz"], instancia["demandas"], instancia["capacidad"]
    if vecinos is None:
        vecinos = vecinos_cercanos(matriz)
    k = min(k_vecinos, instancia["n"] - 1)
    rutas = [list(r) for r in rutas if r]
    movimientos = 0

    def indexar():
        donde = {}
        for r, ruta in enumerate(rutas):
            for p, c in enumerate(ruta):
                donde[c] = (r, p)
        return donde

    def vecinos_en_ruta(ruta, p):
        anterior = ruta[p - 1] if p > 0 else 0
        posterior = ruta[p + 1] if p < len(ruta) - 1 else 0
        return anterior, posterior

    mejora = True
    while mejora:
        mejora = False
        # 2-opt intra-ruta
        for r, ruta in enumerate(rutas):
            nueva = dos_opt_ruta(ruta, matriz)
            if nueva != ruta:
                antes, despues = distancia_ruta(ruta, matriz), distancia_ruta(nueva, matriz)
                if despues < antes - TOLERANCIA:
                    rutas[r] = nueva
                    movimientos += 1
        cargas = [sum(demandas[c] for c in ruta) for ruta in rutas]
        longitudes = [distancia_ruta(ruta, matriz) for ruta in rutas]
        donde = indexar()

        # relocate: mover c junto a (antes o después de) un vecino v de otra ruta
        for c in range(1, instancia["n"] + 1):
            ra, pa = donde[c]
            origen = rutas[ra]
            ant, pos = vecinos_en_ruta(origen, pa)
            ganancia = matriz[ant][c] + matriz[c][pos] - matriz[ant][pos]
            for v in vecinos[c][:k]:
                rb, pb = donde[v]
                if rb == ra or cargas[rb] + demandas[c] > capacidad:
                    continue
                destino = rutas[rb]
                hecho = False
                for insercion in (pb, pb + 1):  # antes de v o después de v
                    previo = destino[insercion - 1] if insercion > 0 else 0
                    proximo = destino[insercion] if insercion < len(destino) else 0
                    costo = matriz[previo][c] + matriz[c][proximo] - matriz[previo][proximo]
                    if costo - ganancia >= -TOLERANCIA:
                        continue
                    if not (_cabe(longitudes[rb] + costo, len(destino) + 1, instancia)
                            and _cabe(longitudes[ra] - ganancia, len(origen) - 1, instancia)):
                        continue
                    origen.pop(pa)
                    destino.insert(insercion, c)
                    cargas[ra] -= demandas[c]
                    cargas[rb] += demandas[c]
                    longitudes[ra] -= ganancia
                    longitudes[rb] += costo
                    for p, x in enumerate(origen):
                        donde[x] = (ra, p)
                    for p, x in enumerate(destino):
                        donde[x] = (rb, p)
                    movimientos += 1
                    mejora = hecho = True
                    break
                if hecho:
                    break

        # swap: intercambiar c con un vecino v de otra ruta
        for c in range(1, instancia["n"] + 1):
            for v in vecinos[c][:k]:
                ra, pa = donde[c]
                rb, pb = donde[v]
                if ra == rb:
                    continue
                if (cargas[ra] - demandas[c] + demandas[v] > capacidad
                        or cargas[rb] - demandas[v] + demandas[c] > capacidad):
                    continue
                ruta_a, ruta_b = rutas[ra], rutas[rb]
                ant_a, pos_a = vecinos_en_ruta(ruta_a, pa)
                ant_b, pos_b = vecinos_en_ruta(ruta_b, pb)
                delta_a = matriz[ant_a][v] + matriz[v][pos_a] - matriz[ant_a][c] - matriz[c][pos_a]
                delta_b = matriz[ant_b][c] + matriz[c][pos_b] - matriz[ant_b][v] - matriz[v][pos_b]
                if delta_a + delta_b >= -TOLERANCIA:
                    continue
                if not (_cabe(longitudes[ra] + delta_a, len(ruta_a), instancia)
                        and _cabe(longitudes[rb] + delta_b, len(ruta_b), instancia)):
                    continue
                ruta_a[pa], ruta_b[pb] = v, c
                cargas[ra] += demandas[v] - demandas[c]
                cargas[rb] += demandas[c] - demandas[v]
                longitudes[ra] += delta_a
                longitudes[rb] += delta_b
                donde[c], donde[v] = (rb, pb), (ra, pa)
                movimientos += 1
                mejora = True
                break

        rutas = [ruta for ruta in rutas if ruta]
    return rutas, movimientos


def resolver_propuesto(instancia, busqueda=True):
    """Clarke-Wright (+ búsqueda local). Devuelve (rutas, movimientos, tiempo en s)."""
    inicio = time.perf_counter()
    rutas = clarke_wright3(instancia)
    movimientos = 0
    if busqueda:
        rutas, movimientos = busqueda_local3(rutas, instancia)
    return rutas, movimientos, time.perf_counter() - inicio

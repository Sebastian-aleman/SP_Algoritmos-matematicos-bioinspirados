"""Búsqueda local para el CVRP: 2-opt intra-ruta, relocate y swap inter-ruta."""

TOLERANCIA = 1e-9


def _dos_opt(rutas, matriz):
    """Invierte un tramo de una ruta si acorta su recorrido."""
    for ruta in rutas:
        sec = [0] + ruta + [0]
        for a in range(1, len(sec) - 2):
            for b in range(a + 1, len(sec) - 1):
                delta = (matriz[sec[a - 1]][sec[b]] + matriz[sec[a]][sec[b + 1]]
                         - matriz[sec[a - 1]][sec[a]] - matriz[sec[b]][sec[b + 1]])
                if delta < -TOLERANCIA:
                    ruta[a - 1:b] = reversed(ruta[a - 1:b])
                    return {"tipo": "2-opt", "nodos": (sec[a], sec[b]), "ahorro": -delta,
                            "detalle": f"invierte el tramo {sec[a]}..{sec[b]}"}
    return None


def _relocate(rutas, matriz, demandas, capacidad):
    """Mueve un cliente a otra ruta (puede dejar vacía la ruta de origen)."""
    for a, origen in enumerate(rutas):
        for p, cliente in enumerate(origen):
            anterior = origen[p - 1] if p > 0 else 0
            posterior = origen[p + 1] if p < len(origen) - 1 else 0
            ganancia = (matriz[anterior][cliente] + matriz[cliente][posterior]
                        - matriz[anterior][posterior])
            for b, destino in enumerate(rutas):
                if a == b or sum(demandas[k] for k in destino) + demandas[cliente] > capacidad:
                    continue
                for pos in range(len(destino) + 1):
                    previo = destino[pos - 1] if pos > 0 else 0
                    proximo = destino[pos] if pos < len(destino) else 0
                    costo = (matriz[previo][cliente] + matriz[cliente][proximo]
                             - matriz[previo][proximo])
                    if costo - ganancia < -TOLERANCIA:
                        origen.pop(p)
                        destino.insert(pos, cliente)
                        return {"tipo": "relocate", "nodos": (cliente,),
                                "ahorro": ganancia - costo,
                                "detalle": f"mueve {cliente} entre {previo} y {proximo}"}
    return None


def _swap(rutas, matriz, demandas, capacidad):
    """Intercambia dos clientes de rutas distintas."""
    cargas = [sum(demandas[k] for k in ruta) for ruta in rutas]
    for a in range(len(rutas)):
        for b in range(a + 1, len(rutas)):
            ruta_a, ruta_b = rutas[a], rutas[b]
            for p, v in enumerate(ruta_a):
                for q, w in enumerate(ruta_b):
                    if (cargas[a] - demandas[v] + demandas[w] > capacidad
                            or cargas[b] - demandas[w] + demandas[v] > capacidad):
                        continue
                    ant_a = ruta_a[p - 1] if p > 0 else 0
                    pos_a = ruta_a[p + 1] if p < len(ruta_a) - 1 else 0
                    ant_b = ruta_b[q - 1] if q > 0 else 0
                    pos_b = ruta_b[q + 1] if q < len(ruta_b) - 1 else 0
                    delta = (matriz[ant_a][w] + matriz[w][pos_a]
                             + matriz[ant_b][v] + matriz[v][pos_b]
                             - matriz[ant_a][v] - matriz[v][pos_a]
                             - matriz[ant_b][w] - matriz[w][pos_b])
                    if delta < -TOLERANCIA:
                        ruta_a[p], ruta_b[q] = w, v
                        return {"tipo": "swap", "nodos": (v, w), "ahorro": -delta,
                                "detalle": f"intercambia {v} y {w}"}
    return None


def busqueda_local(rutas, matriz, demandas, capacidad):
    """Aplica 2-opt, relocate y swap con primera mejora hasta un óptimo local.

    Tras cada movimiento aplicado se reinicia la exploración. Devuelve
    (rutas, movimientos); cada movimiento tiene tipo, nodos, ahorro y detalle.
    """
    rutas = [list(ruta) for ruta in rutas]
    movimientos = []
    while True:
        movimiento = (_dos_opt(rutas, matriz)
                      or _relocate(rutas, matriz, demandas, capacidad)
                      or _swap(rutas, matriz, demandas, capacidad))
        if movimiento is None:
            break
        movimientos.append(movimiento)
        rutas = [ruta for ruta in rutas if ruta]
    return rutas, movimientos

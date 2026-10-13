"""Evaluación y reporte de una solución CVRP."""

import json
import math


def evaluar_solucion(rutas, matriz, demandas, capacidad):
    """Calcula las métricas por ruta y las globales de una solución."""
    detalle = []
    for numero, ruta in enumerate(rutas, start=1):
        secuencia = [0] + list(ruta) + [0]
        carga = sum(demandas[k] for k in ruta)
        distancia = sum(matriz[a][b] for a, b in zip(secuencia, secuencia[1:]))
        detalle.append({
            "ruta": numero,
            "secuencia": secuencia,
            "carga": carga,
            "distancia": distancia,
            "desperdicio_pct": (capacidad - carga) / capacidad * 100,
        })

    demanda_total = sum(demandas)
    return {
        "rutas": detalle,
        "distancia_total": sum(r["distancia"] for r in detalle),
        "numero_rutas": len(detalle),
        "demanda_total": demanda_total,
        "capacidad": capacidad,
        "vehiculos_minimos": math.ceil(demanda_total / capacidad),
    }


def guardar_solucion_json(solucion, ruta):
    with open(ruta, "w", encoding="utf-8") as archivo:
        json.dump(solucion, archivo, indent=2, ensure_ascii=False)


def guardar_solucion_tex(solucion, ruta):
    """Guarda la tabla resumen de la solución."""
    q = solucion["capacidad"]
    lineas = [
        r"\begin{tabular}{clrrr}",
        r"\toprule",
        r"Ruta & Secuencia & Carga & Distancia & Espacio desperdiciado (\%) \\",
        r"\midrule",
    ]
    for r in solucion["rutas"]:
        secuencia = "-".join(map(str, r["secuencia"]))
        lineas.append(
            f"{r['ruta']} & {secuencia} & {r['carga']} & {r['distancia']:.2f} & "
            f"{r['desperdicio_pct']:.2f} \\\\"
        )
    carga_total = solucion["demanda_total"]
    desperdicio_total = (solucion["numero_rutas"] * q - carga_total) / (solucion["numero_rutas"] * q) * 100
    lineas += [
        r"\midrule",
        f"Total & {solucion['numero_rutas']} rutas & {carga_total} & "
        f"{solucion['distancia_total']:.2f} & {desperdicio_total:.2f} \\\\",
        r"\midrule",
        rf"\multicolumn{{5}}{{l}}{{Capacidad $Q={q}$; demanda total $={carga_total}$; "
        rf"vehículos mínimos $=\lceil {carga_total}/{q} \rceil = {solucion['vehiculos_minimos']}$}} \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    with open(ruta, "w", encoding="utf-8") as archivo:
        archivo.write("\n".join(lineas) + "\n")

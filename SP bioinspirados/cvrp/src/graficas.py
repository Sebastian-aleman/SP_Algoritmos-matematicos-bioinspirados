"""Gráficas de la instancia y de la solución."""

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

COLORES_RUTAS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                 "#e87ba4", "#008300", "#6250d6"]
COLOR_DEPOSITO = "#d03b3b"
COLOR_TEXTO = "#0b0b0b"
COLOR_SECUNDARIO = "#52514e"


def _nuevo_eje(coordenadas):
    figura, eje = plt.subplots(figsize=(7.5, 6.5))
    xs = [x for x, _ in coordenadas]
    ys = [y for _, y in coordenadas]
    eje.set_xlim(min(xs) - 10, max(xs) + 14)
    eje.set_ylim(min(ys) - 8, max(ys) + 8)
    eje.set_aspect("equal")
    eje.set_xlabel("x", color=COLOR_SECUNDARIO)
    eje.set_ylabel("y", color=COLOR_SECUNDARIO)
    eje.grid(True, color="#e4e3df", linewidth=0.6)
    eje.set_axisbelow(True)
    eje.tick_params(colors=COLOR_SECUNDARIO)
    for borde in eje.spines.values():
        borde.set_color("#c9c8c2")
    return figura, eje


def _dibujar_nodos(eje, coordenadas, demandas, color_clientes):
    """color_clientes: lista con el color de cada nodo (se ignora la posición 0)."""
    x0, y0 = coordenadas[0]
    for i, (x, y) in enumerate(coordenadas[1:], start=1):
        eje.scatter(x, y, s=70, color=color_clientes[i], edgecolor="white",
                    linewidth=1.2, zorder=3)
        # La etiqueta se aleja del depósito para no chocar con los arcos de la ruta
        dx, dy = x - x0, y - y0
        norma = math.hypot(dx, dy) or 1.0
        eje.annotate(f"{i} (q={demandas[i]})", (x, y),
                     xytext=(11 * dx / norma, 11 * dy / norma),
                     textcoords="offset points", fontsize=9, color=COLOR_TEXTO,
                     ha="left" if dx >= 0 else "right",
                     va="bottom" if dy >= 0 else "top")
    eje.scatter(x0, y0, s=130, marker="s", color=COLOR_DEPOSITO, edgecolor="white",
                linewidth=1.2, zorder=4)


def graficar_nodos(instancia, ruta):
    coordenadas, demandas = instancia["coordenadas"], instancia["demandas"]
    figura, eje = _nuevo_eje(coordenadas)
    _dibujar_nodos(eje, coordenadas, demandas, [COLORES_RUTAS[0]] * len(coordenadas))
    eje.set_title(f"{instancia['nombre']}: nodos (Q={instancia['capacidad']})",
                  color=COLOR_TEXTO)
    eje.legend(handles=[
        Line2D([], [], marker="s", linestyle="", color=COLOR_DEPOSITO,
               markersize=9, label="Depósito (0)"),
        Line2D([], [], marker="o", linestyle="", color=COLORES_RUTAS[0],
               markersize=8, label="Cliente i (q = demanda)"),
    ], loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
    figura.savefig(ruta, bbox_inches="tight")
    plt.close(figura)


def _formato_tiempo(segundos):
    if segundos >= 1:
        return f"{segundos:.2f} s"
    if segundos >= 1e-3:
        return f"{segundos * 1e3:.1f} ms"
    return f"{segundos * 1e6:.0f} µs"


def graficar_comparacion(filas, ruta):
    """Dos gráficas de barras: Z por método y tiempo por método (escala log)."""
    nombres = [f["metodo"].replace(" (", "\n(").replace(" + ", "\n+ ") for f in filas]
    figura, (eje_z, eje_t) = plt.subplots(1, 2, figsize=(11, 4.6))

    paneles = [
        (eje_z, [f["z"] for f in filas], "Distancia total Z", lambda v: f"{v:.2f}"),
        (eje_t, [f["tiempo"] for f in filas], "Tiempo (s, escala log)", _formato_tiempo),
    ]
    for eje, valores, titulo, formato in paneles:
        barras = eje.bar(nombres, valores, width=0.5, color=COLORES_RUTAS[0])
        eje.bar_label(barras, labels=[formato(v) for v in valores], padding=3,
                      fontsize=9, color=COLOR_TEXTO)
        eje.set_title(titulo, color=COLOR_TEXTO)
        eje.grid(True, axis="y", color="#e4e3df", linewidth=0.6)
        eje.set_axisbelow(True)
        eje.tick_params(colors=COLOR_SECUNDARIO)
        for lado in ("top", "right"):
            eje.spines[lado].set_visible(False)
        for lado in ("left", "bottom"):
            eje.spines[lado].set_color("#c9c8c2")

    eje_z.set_ylim(0, max(f["z"] for f in filas) * 1.12)
    eje_t.set_yscale("log")
    tiempos = [f["tiempo"] for f in filas]
    eje_t.set_ylim(min(tiempos) / 3, max(tiempos) * 4)

    figura.tight_layout()
    figura.savefig(ruta, bbox_inches="tight")
    plt.close(figura)


def graficar_rutas(instancia, solucion, ruta, titulo="Rutas"):
    coordenadas, demandas = instancia["coordenadas"], instancia["demandas"]
    if len(solucion["rutas"]) > len(COLORES_RUTAS):
        raise ValueError("Hay más rutas que colores disponibles")
    figura, eje = _nuevo_eje(coordenadas)
    color_clientes = [COLOR_SECUNDARIO] * len(coordenadas)
    leyenda = [Line2D([], [], marker="s", linestyle="", color=COLOR_DEPOSITO,
                      markersize=9, label="Depósito (0)")]

    for r, color in zip(solucion["rutas"], COLORES_RUTAS):
        secuencia = r["secuencia"]
        for a, b in zip(secuencia, secuencia[1:]):
            eje.annotate("", xy=coordenadas[b], xytext=coordenadas[a], zorder=2,
                         arrowprops=dict(arrowstyle="-|>", color=color, linewidth=2,
                                         mutation_scale=16, shrinkA=6, shrinkB=6))
        for k in secuencia[1:-1]:
            color_clientes[k] = color
        etiqueta = (f"Ruta {r['ruta']}: {'-'.join(map(str, secuencia))}  |  "
                    f"distancia = {r['distancia']:.2f}, "
                    f"carga = {r['carga']}/{solucion['capacidad']}")
        leyenda.append(Line2D([], [], color=color, linewidth=2, marker="o",
                              markersize=6, label=etiqueta))

    _dibujar_nodos(eje, coordenadas, demandas, color_clientes)
    eje.set_title(f"{titulo} (distancia total = {solucion['distancia_total']:.2f})",
                  color=COLOR_TEXTO)
    eje.legend(handles=leyenda, loc="upper center", bbox_to_anchor=(0.5, -0.1),
               frameon=False, fontsize=9)
    figura.savefig(ruta, bbox_inches="tight")
    plt.close(figura)

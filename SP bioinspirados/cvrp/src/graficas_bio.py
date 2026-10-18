"""Gráficas de los algoritmos bioinspirados (convergencia del GA y del ACO)."""

import statistics

from src.graficas import COLOR_SECUNDARIO, COLOR_TEXTO, COLORES_RUTAS, plt


def _rellenar(historiales):
    """Iguala la longitud de los historiales repitiendo el último valor de cada uno."""
    largo = max(len(h) for h in historiales)
    return [list(h) + [h[-1]] * (largo - len(h)) for h in historiales]


def _por_posicion(historiales, funcion):
    return [funcion(valores) for valores in zip(*historiales)]


def graficar_convergencia(paneles, z_opt, ruta):
    """Figura con un panel por algoritmo.

    Cada panel es un dict con titulo, eje_x, inicio (número de la primera
    generación/iteración del historial), historial_mejor e historial_promedio
    (una lista por corrida). Dibuja la mediana del mejor con una banda mínimo-máximo,
    la mediana del promedio y Z* punteado. Las corridas que paran antes se
    rellenan con su último valor.
    """
    figura, ejes = plt.subplots(1, len(paneles), figsize=(11, 4.6))
    for eje, panel in zip(ejes, paneles):
        mejores = _rellenar(panel["historial_mejor"])
        promedios = _rellenar(panel["historial_promedio"])
        x = range(panel["inicio"], panel["inicio"] + len(mejores[0]))

        eje.fill_between(x, _por_posicion(mejores, min), _por_posicion(mejores, max),
                         color=COLORES_RUTAS[0], alpha=0.18, linewidth=0,
                         label="Mejor: mínimo–máximo")
        eje.plot(x, _por_posicion(mejores, statistics.median), color=COLORES_RUTAS[0],
                 linewidth=2, label="Mejor: mediana")
        eje.plot(x, _por_posicion(promedios, statistics.median), color=COLORES_RUTAS[1],
                 linewidth=1.6, label="Promedio de la población: mediana")
        eje.axhline(z_opt, color=COLOR_TEXTO, linestyle=":", linewidth=1.4,
                    label=f"$Z^*$ = {z_opt:.2f}")

        eje.set_title(f"{panel['titulo']} ({len(mejores)} corridas)", color=COLOR_TEXTO)
        eje.set_xlabel(panel["eje_x"], color=COLOR_SECUNDARIO)
        eje.set_ylabel("Distancia total Z", color=COLOR_SECUNDARIO)
        eje.set_xlim(x[0], x[-1])
        eje.grid(True, color="#e4e3df", linewidth=0.6)
        eje.set_axisbelow(True)
        eje.tick_params(colors=COLOR_SECUNDARIO)
        for lado in ("top", "right"):
            eje.spines[lado].set_visible(False)
        for lado in ("left", "bottom"):
            eje.spines[lado].set_color("#c9c8c2")

    manejadores, etiquetas = ejes[0].get_legend_handles_labels()
    figura.legend(manejadores, etiquetas, loc="lower center", ncol=len(etiquetas),
                  frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.04))
    figura.tight_layout(rect=(0, 0.06, 1, 1))
    figura.savefig(ruta, bbox_inches="tight")
    plt.close(figura)

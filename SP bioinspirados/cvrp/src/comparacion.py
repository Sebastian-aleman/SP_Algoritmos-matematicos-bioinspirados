"""Comparación entre el modelo exacto y los métodos heurísticos."""


def desperdicio_global(solucion):
    """% de capacidad no usada sobre el total de la flota empleada."""
    capacidad_flota = solucion["numero_rutas"] * solucion["capacidad"]
    return (capacidad_flota - solucion["demanda_total"]) / capacidad_flota * 100


def construir_comparacion(metodos):
    """metodos: lista de (nombre, solucion, tiempo); el primero es el óptimo.

    GAP_costo(%) = (Z_metodo - Z_opt) / Z_opt * 100
    Diferencia de tiempo(%) = (t_metodo - t_opt) / t_opt * 100
    """
    _, solucion_opt, t_opt = metodos[0]
    z_opt = solucion_opt["distancia_total"]
    filas = []
    for nombre, solucion, tiempo in metodos:
        z = solucion["distancia_total"]
        filas.append({
            "metodo": nombre,
            "z": z,
            "numero_rutas": solucion["numero_rutas"],
            "desperdicio_global_pct": desperdicio_global(solucion),
            "tiempo": tiempo,
            # round + 0.0 evita un "-0.00" por error de punto flotante
            "gap_costo_pct": round((z - z_opt) / z_opt * 100, 9) + 0.0,
            "diferencia_tiempo_pct": (tiempo - t_opt) / t_opt * 100,
        })
    return filas


def guardar_comparacion_tex(filas, ruta):
    lineas = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Método & $Z$ & N.º rutas & Desperdicio global (\%) & Tiempo (s) & "
        r"GAP costo (\%) & Dif. tiempo (\%) \\",
        r"\midrule",
    ]
    for f in filas:
        lineas.append(
            f"{f['metodo']} & {f['z']:.2f} & {f['numero_rutas']} & "
            f"{f['desperdicio_global_pct']:.2f} & {f['tiempo']:.6f} & "
            f"{f['gap_costo_pct']:.2f} & {f['diferencia_tiempo_pct']:.4f} \\\\"
        )
    lineas += [r"\bottomrule", r"\end{tabular}"]
    with open(ruta, "w", encoding="utf-8") as archivo:
        archivo.write("\n".join(lineas) + "\n")

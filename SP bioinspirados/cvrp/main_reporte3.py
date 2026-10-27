"""Tablas (.tex) y figuras (.pdf) del Entregable 3 a partir de resultados/etapa3/corridas.jsonl.

Uso, desde la carpeta cvrp/:
    python main_reporte3.py

Salidas:
    resultados/etapa3/tablas/*.tex   cuerpos de tabla (tabular) para \\input en el reporte
    figuras/etapa3/*.pdf             figuras del reporte
    resultados/etapa3/resumen.json   cifras citadas en el texto
"""

import json
import math
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

from src.experimentos3 import (FACTORES_ACO, FACTORES_GA, configuraciones,
                               leer_corridas, seleccionar_mejores, tiempo_max)
from src.instancias_cvrplib import INSTANCIAS, INSTANCIAS_CMT, INSTANCIAS_X, cargar

RAIZ = Path(__file__).resolve().parent
ETAPA = RAIZ / "resultados" / "etapa3"
TABLAS = ETAPA / "tablas"
FIGURAS = RAIZ / "figuras" / "etapa3"
DATOS = RAIZ / "datos"

METODOS = ["CW", "CW+BL", "GA", "ACO", "MEM"]
NOMBRE = {"CW": "Clarke--Wright", "CW+BL": "CW + búsqueda local", "GA": "GA",
          "ACO": "ACO", "MEM": "GA memético"}
NOMBRE_FIG = {**NOMBRE, "CW": "Clarke–Wright"}
COLOR = {"CW": "#9e9e9e", "CW+BL": "#f0a30a", "GA": "#2b7bd6", "ACO": "#e8643a",
         "MEM": "#1aa37a"}
CONFIG_FINAL = {"GA": "GA-final", "ACO": "ACO-final", "MEM": "GA-memetico"}
SUBINSTANCIAS = [f"CMT1-k{k}" for k in (10, 15, 20, 25, 30, 40)]
TOL_BKS = 0.005  # GAP (%) por debajo del cual se considera que se alcanzó la BKS

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.bbox": "tight"})


# --- utilidades ---

def num(x, d=2):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    if abs(x) < 0.5 * 10 ** (-d):
        x = 0.0
    return f"{x:,.{d}f}"


def corto(inst):
    return inst.replace("X-", "")


def escribir(nombre, lineas):
    TABLAS.mkdir(parents=True, exist_ok=True)
    (TABLAS / nombre).write_text("\n".join(lineas) + "\n", encoding="utf-8")


def negrita_min(valores, d=2, mayor=False):
    """Formatea una columna marcando en negritas el mejor valor."""
    validos = [v for v in valores if v is not None]
    mejor = (max if mayor else min)(validos) if validos else None
    salida = []
    for v in valores:
        texto = num(v, d)
        if v is not None and mejor is not None and abs(v - mejor) < 10 ** (-d) / 2:
            texto = r"\textbf{" + texto + "}"
        salida.append(texto)
    return salida


def media(xs):
    xs = [x for x in xs if x is not None]
    return st.mean(xs) if xs else None


# --- carga ---

def cargar_datos():
    registros = [r for r in leer_corridas(ETAPA / "corridas.jsonl") if "error" not in r]
    fac = [r for r in registros if r["fase"] == "factorial"]
    fin = [r for r in registros if r["fase"] == "final"]
    prop = {(r["config"], r["instancia"]): r for r in registros if r["fase"] == "propuesto"}
    exa = {r["instancia"]: r for r in registros if r["fase"] == "exacto"}
    return registros, fac, fin, prop, exa


def por_instancia(fin, prop, metodo, inst, campo="gap_bks_pct"):
    """Promedio de un campo para un método en una instancia (CW y CW+BL son deterministas)."""
    if metodo in ("CW", "CW+BL"):
        return prop[metodo, inst].get(campo)
    filas = [r for r in fin if r["config"] == CONFIG_FINAL[metodo] and r["instancia"] == inst]
    return media([r.get(campo) for r in filas])


# --- tablas ---

def tabla_instancias():
    filas = []
    for nombre in INSTANCIAS:
        i = cargar(nombre, DATOS)
        kmin = math.ceil(sum(i["demandas"]) / i["capacidad"])
        bks = num(i["bks"], 2) if not i["redondeo"] else f"{i['bks']:,.0f}"
        filas.append(" & ".join([
            nombre, str(i["n"]), str(i["capacidad"]), str(kmin),
            num(i["limite_ruta"], 0) if i["limite_ruta"] else "--",
            num(i["tiempo_servicio"], 0) if i["limite_ruta"] else "--",
            "nint" if i["redondeo"] else "real", bks, num(tiempo_max(i["n"]), 1)]) + r" \\")
    escribir("instancias.tex", [
        r"\begin{tabular}{lrrrrrlrr}", r"\toprule",
        r"Instancia & $n$ & $Q$ & $K_{\min}$ & \makecell{Duración\\máx.} & Servicio & "
        r"Distancia & BKS & \makecell{Presupuesto\\GA/ACO (s)} \\", r"\midrule",
        *filas[:14], r"\midrule", *filas[14:], r"\bottomrule", r"\end{tabular}"])


def tabla_exacto(exa, resumen):
    orden = SUBINSTANCIAS + ["CMT1", "CMT2", "CMT3", "CMT12", "CMT11", "CMT4", "CMT5"]
    texto_estatus = {"optimo": "óptimo", "factible": "factible", "sin_solucion": "sin solución"}
    filas = []
    for inst in orden:
        r = exa[inst]
        filas.append(" & ".join([
            inst, str(r["n"]), f"{r['numero_variables']:,}", f"{r['numero_restricciones']:,}",
            texto_estatus.get(r["estatus"], r["estatus"]), num(r["z"]), num(r["cota_inferior"]),
            num(r["gap_mip_pct"]), num(r["bks"]) if r["bks"] else "--",
            num(r.get("gap_bks_pct")), f"{r['nodos']:,}", num(r["tiempo"], 1)]) + r" \\")
        if inst == "CMT1-k40":
            filas.append(r"\midrule")
    escribir("exacto.tex", [
        r"\begin{tabular}{lrrrlrrrrrrr}", r"\toprule",
        r"Instancia & $n$ & Variables & Restricciones & Estatus & $Z$ & \makecell{Cota\\inferior} & "
        r"\makecell{Brecha\\MIP (\%)} & BKS & \makecell{GAP vs\\BKS (\%)} & Nodos & Tiempo (s) \\",
        r"\midrule", *filas, r"\bottomrule", r"\end{tabular}"])
    resumen["exacto"] = {k: {c: exa[k][c] for c in ("estatus", "z", "cota_inferior", "tiempo",
                                                     "nodos", "gap_mip_pct")} for k in orden}


def tabla_gap_exacto(exa, fin, prop, resumen):
    orden = SUBINSTANCIAS + ["CMT1", "CMT2", "CMT3", "CMT12"]
    filas, datos = [], {}
    for inst in orden:
        z = exa[inst]["z"]
        gaps = []
        for m in METODOS:
            costo = por_instancia(fin, prop, m, inst, "costo")
            gaps.append((costo - z) / z * 100)
        datos[inst] = dict(zip(METODOS, gaps))
        marca = "" if exa[inst]["estatus"] == "optimo" else r"$^\dagger$"
        filas.append(" & ".join([inst, str(exa[inst]["n"]), num(z) + marca,
                                 *negrita_min(gaps)]) + r" \\")
        if inst == "CMT1-k30":
            filas.append(r"\midrule")
    escribir("gap_exacto.tex", [
        r"\begin{tabular}{lrrrrrrr}", r"\toprule",
        r"Instancia & $n$ & $Z$ exacto & \makecell{Clarke--\\Wright} & \makecell{CW +\\búsq. local} & "
        r"GA & ACO & \makecell{GA\\memético} \\", r"\midrule", *filas,
        r"\bottomrule", r"\end{tabular}"])
    resumen["gap_exacto"] = datos
    return datos


def resumen_factorial(fac, algoritmo):
    filas = [r for r in fac if r["algoritmo"] == algoritmo]
    mejores, efectos, rango, _ = seleccionar_mejores(filas, algoritmo)
    costo = defaultdict(list)
    for r in filas:
        costo[r["config"], r["instancia"]].append(r["costo"])
    ganadas = Counter()
    for inst in INSTANCIAS:
        prom = {c: st.mean(costo[c, inst]) for c, _ in configuraciones(algoritmo)}
        mejor = min(prom.values())
        ganadas.update(c for c, v in prom.items() if v - mejor < 1e-9)
    tabla = {}
    for cid, niveles in configuraciones(algoritmo):
        f = [r for r in filas if r["config"] == cid]
        tabla[cid] = {
            "niveles": niveles, "rango": rango[cid], "ganadas": ganadas[cid],
            "gap": media(r["gap_bks_pct"] for r in f),
            "gap_cmt": media(r["gap_bks_pct"] for r in f if r["instancia"] in INSTANCIAS_CMT),
            "gap_x": media(r["gap_bks_pct"] for r in f if r["instancia"] in INSTANCIAS_X),
            "tiempo": media(r["tiempo"] for r in f),
            "t_mejor": media(r["tiempo_hasta_mejor"] for r in f),
            "it": media(r["iteraciones"] for r in f),
            "it_mejor": media(r["iteracion_mejor"] for r in f),
            "paciencia": 100 * sum(r["motivo_paro"] == "paciencia" for r in f) / len(f),
        }
    return mejores, efectos, tabla


def tabla_factorial(algoritmo, tabla, archivo):
    orden = sorted(tabla, key=lambda c: tabla[c]["rango"])
    columnas = ["rango", "ganadas", "gap", "gap_cmt", "gap_x", "tiempo", "t_mejor", "it",
                "it_mejor", "paciencia"]
    decimales = {"rango": 2, "ganadas": 0, "tiempo": 1, "t_mejor": 1, "it": 0, "it_mejor": 0,
                 "paciencia": 1}
    mayor = {"ganadas"}
    sin_negrita = {"it", "it_mejor", "paciencia"}
    formato = {}
    for col in columnas:
        vals = [tabla[c][col] for c in orden]
        d = decimales.get(col, 2)
        formato[col] = ([num(v, d) for v in vals] if col in sin_negrita
                        else negrita_min(vals, d, col in mayor))
    filas = []
    for k, c in enumerate(orden):
        niv = tabla[c]["niveles"]
        if algoritmo == "GA":
            etiquetas = [str(niv["puntos_cruce"]), f"{niv['prob_mutacion']:.2f}", niv["seleccion"]]
        else:
            etiquetas = [str(niv["alfa"]), str(niv["beta"]), f"{niv['rho']:.1f}"]
        filas.append(" & ".join(etiquetas + [formato[col][k] for col in columnas]) + r" \\")
    if algoritmo == "GA":
        cabeza = r"Cruce & $p_m$ & Selección"
        iters = r"\makecell{Gen.\\totales} & \makecell{Gen. del\\mejor}"
    else:
        cabeza = r"$\alpha$ & $\beta$ & $\rho$"
        iters = r"\makecell{It.\\totales} & \makecell{It. del\\mejor}"
    escribir(archivo, [
        r"\begin{tabular}{lll" + "r" * len(columnas) + "}", r"\toprule",
        cabeza + r" & \makecell{Rango\\prom.} & \makecell{Inst.\\ganadas} & \makecell{GAP\\(\%)} & "
        r"\makecell{GAP\\CMT (\%)} & \makecell{GAP\\X (\%)} & \makecell{Tiempo\\(s)} & "
        r"\makecell{Tiempo al\\mejor (s)} & " + iters + r" & \makecell{Paro por\\paciencia (\%)} \\",
        r"\midrule", *filas, r"\bottomrule", r"\end{tabular}"])


def tabla_efectos(algoritmo, tabla, efectos, mejores, archivo):
    factores = FACTORES_GA if algoritmo == "GA" else FACTORES_ACO
    nombre = {"puntos_cruce": "Puntos de cruce", "prob_mutacion": r"Prob. de mutación $p_m$",
              "seleccion": "Selección de padres", "alfa": r"Peso de la feromona $\alpha$",
              "beta": r"Peso de la visibilidad $\beta$", "rho": r"Persistencia $\rho$"}
    filas = []
    for factor, niveles in factores.items():
        for k, nivel in enumerate(niveles):
            usados = [t for t in tabla.values() if t["niveles"][factor] == nivel]
            valores = [efectos[factor][nivel], media(t["gap"] for t in usados),
                       media(t["tiempo"] for t in usados), media(t["t_mejor"] for t in usados),
                       media(t["it"] for t in usados), media(t["it_mejor"] for t in usados)]
            celdas = [num(valores[0]), num(valores[1]), num(valores[2], 1), num(valores[3], 1),
                      num(valores[4], 0), num(valores[5], 0)]
            texto_nivel = f"{nivel:g}" if isinstance(nivel, float) else str(nivel)
            if nivel == mejores[factor]:
                texto_nivel = r"\textbf{" + texto_nivel + "}"
                celdas[0] = r"\textbf{" + celdas[0] + "}"
            filas.append(" & ".join([nombre[factor] if k == 0 else "", texto_nivel] + celdas)
                         + r" \\")
        filas.append(r"\addlinespace")
    filas.pop()
    it = "Gen." if algoritmo == "GA" else "It."
    escribir(archivo, [
        r"\begin{tabular}{llrrrrrr}", r"\toprule",
        r"Factor & Nivel & \makecell{Rango\\prom.} & GAP (\%) & Tiempo (s) & "
        r"\makecell{Tiempo al\\mejor (s)} & \makecell{" + it + r"\\totales} & \makecell{" + it +
        r" del\\mejor} \\", r"\midrule", *filas, r"\bottomrule", r"\end{tabular}"])


def tabla_confirmacion(fac, fin, tablas, mejores, resumen):
    filas, datos = [], {}
    for alg in ("GA", "ACO"):
        cid = next(c for c, niv in configuraciones(alg) if niv == mejores[alg])
        t = tablas[alg][cid]
        f = [r for r in fin if r["config"] == CONFIG_FINAL[alg] and r["instancia"] in INSTANCIAS]
        final = {"gap": media(r["gap_bks_pct"] for r in f),
                 "gap_cmt": media(r["gap_bks_pct"] for r in f if r["instancia"] in INSTANCIAS_CMT),
                 "gap_x": media(r["gap_bks_pct"] for r in f if r["instancia"] in INSTANCIAS_X),
                 "tiempo": media(r["tiempo"] for r in f), "it": media(r["iteraciones"] for r in f)}
        promedio_todas = media(x["gap"] for x in tablas[alg].values())
        datos[alg] = {"config": cid, "factorial": t["gap"], "final": final["gap"],
                      "promedio_factorial": promedio_todas}
        for k, (fase, sem, v) in enumerate((("Factorial", "0--2", t), ("Final", "100--104", final))):
            filas.append(" & ".join([alg if k == 0 else "", fase, sem, num(v["gap"]),
                                     num(v["gap_cmt"]), num(v["gap_x"]), num(v["tiempo"], 1),
                                     num(v["it"], 0)]) + r" \\")
    escribir("confirmacion.tex", [
        r"\begin{tabular}{lllrrrrr}", r"\toprule",
        r"Algoritmo & Fase & Semillas & GAP (\%) & GAP CMT (\%) & GAP X (\%) & Tiempo (s) & "
        r"Iteraciones \\", r"\midrule", *filas, r"\bottomrule", r"\end{tabular}"])
    resumen["confirmacion"] = datos


def tabla_gap_bks(fin, prop, resumen):
    filas, matriz = [], {m: [] for m in METODOS}
    mejores_corridas = {m: [] for m in ("GA", "ACO", "MEM")}
    for inst in INSTANCIAS:
        bks = prop["CW", inst]["bks"]
        prom = [por_instancia(fin, prop, m, inst) for m in METODOS]
        for m, v in zip(METODOS, prom):
            matriz[m].append(v)
        mejor = []
        for m in ("GA", "ACO", "MEM"):
            v = min(r["gap_bks_pct"] for r in fin
                    if r["config"] == CONFIG_FINAL[m] and r["instancia"] == inst)
            mejores_corridas[m].append(v)
            mejor.append(v)
        bks_txt = f"{bks:,.0f}" if inst.startswith("X-") else num(bks)
        filas.append(" & ".join([inst, str(prop["CW", inst]["n"]), bks_txt, *negrita_min(prom),
                                 *[num(max(v, 0.0)) for v in mejor]]) + r" \\")
        if inst == "CMT14":
            filas.append(r"\midrule")
    filas.append(r"\midrule")
    for etiqueta, sl in (("Promedio CMT", slice(0, 14)), ("Promedio X", slice(14, 24)),
                         ("Promedio", slice(0, 24))):
        prom = [st.mean(matriz[m][sl]) for m in METODOS]
        mej = [st.mean(mejores_corridas[m][sl]) for m in ("GA", "ACO", "MEM")]
        filas.append(" & ".join([etiqueta, "", "", *negrita_min(prom), *[num(v) for v in mej]])
                     + r" \\")
    escribir("gap_bks.tex", [
        r"\begin{tabular}{lrrrrrrrrrr}", r"\toprule",
        r" & & & \multicolumn{5}{c}{GAP promedio vs BKS (\%)} & "
        r"\multicolumn{3}{c}{GAP de la mejor corrida (\%)} \\",
        r"\cmidrule(lr){4-8}\cmidrule(lr){9-11}",
        r"Instancia & $n$ & BKS & CW & CW+BL & GA & ACO & GA mem. & GA & ACO & GA mem. \\",
        r"\midrule", *filas, r"\bottomrule", r"\end{tabular}"])
    resumen["gap_bks"] = {m: {"promedio": st.mean(v), "cmt": st.mean(v[:14]),
                              "x": st.mean(v[14:]), "max": max(v)} for m, v in matriz.items()}
    return matriz


def tabla_wilcoxon(matriz, resumen):
    pares = [("ACO", "GA"), ("CW+BL", "GA"), ("MEM", "CW+BL"), ("MEM", "ACO"), ("MEM", "GA"),
             ("ACO", "CW+BL"), ("CW+BL", "CW")]
    filas, datos = [], []
    for a, b in pares:
        x, y = matriz[a], matriz[b]
        gana = sum(p < q for p, q in zip(x, y))
        prueba = wilcoxon(x, y)
        p = prueba.pvalue
        p_txt = r"$<0.001$" if p < 0.001 else f"{p:.3f}"
        filas.append(" & ".join([f"{NOMBRE[a]} vs {NOMBRE[b]}", f"{gana} / 24",
                                 num(st.mean(x) - st.mean(y)), p_txt,
                                 "sí" if p < 0.05 else "no"]) + r" \\")
        datos.append({"a": a, "b": b, "gana": gana, "p": p})
    escribir("wilcoxon.tex", [
        r"\begin{tabular}{lrrrc}", r"\toprule",
        r"Comparación (A vs B) & \makecell{A mejor en} & \makecell{Diferencia de\\GAP prom. (pp)} & "
        r"$p$ & \makecell{Significativa\\($\alpha = 0.05$)} \\", r"\midrule", *filas,
        r"\bottomrule", r"\end{tabular}"])
    resumen["wilcoxon"] = datos


def tabla_tiempos(fin, prop, resumen):
    filas = []
    acumulado = defaultdict(list)
    for inst in INSTANCIAS:
        valores = [prop["CW", inst]["tiempo"] * 1000, prop["CW+BL", inst]["tiempo"] * 1000]
        for m in ("GA", "ACO", "MEM"):
            valores += [por_instancia(fin, prop, m, inst, "tiempo"),
                        por_instancia(fin, prop, m, inst, "tiempo_hasta_mejor"),
                        por_instancia(fin, prop, m, inst, "iteraciones")]
        for k, v in enumerate(valores):
            acumulado[k].append(v)
        filas.append(" & ".join([inst] + [num(v, 0 if k in (4, 7, 10) else 1)
                                          for k, v in enumerate(valores)]) + r" \\")
        if inst == "CMT14":
            filas.append(r"\midrule")
    filas.append(r"\midrule")
    filas.append(" & ".join(["Promedio"] + [num(st.mean(acumulado[k]), 0 if k in (4, 7, 10) else 1)
                                            for k in range(11)]) + r" \\")
    escribir("tiempos.tex", [
        r"\begin{tabular}{lrrrrrrrrrrr}", r"\toprule",
        r" & CW & CW+BL & \multicolumn{3}{c}{GA} & \multicolumn{3}{c}{ACO} & "
        r"\multicolumn{3}{c}{GA memético} \\",
        r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}\cmidrule(lr){10-12}",
        r"Instancia & (ms) & (ms) & $t$ (s) & $t^*$ (s) & Gen. & $t$ (s) & $t^*$ (s) & It. & "
        r"$t$ (s) & $t^*$ (s) & Gen. \\", r"\midrule", *filas, r"\bottomrule", r"\end{tabular}"])
    resumen["tiempos_promedio"] = {"CW_ms": st.mean(acumulado[0]), "CWBL_ms": st.mean(acumulado[1]),
                                   "GA": st.mean(acumulado[2]), "GA_t*": st.mean(acumulado[3]),
                                   "GA_it": st.mean(acumulado[4]), "ACO": st.mean(acumulado[5]),
                                   "ACO_t*": st.mean(acumulado[6]), "ACO_it": st.mean(acumulado[7]),
                                   "MEM": st.mean(acumulado[8]), "MEM_t*": st.mean(acumulado[9]),
                                   "MEM_it": st.mean(acumulado[10])}


def tabla_inicializacion(fin, resumen):
    instancias = ["CMT1", "CMT13", "CMT5", "X-n280-k17", "X-n459-k26"]
    filas, datos = [], {}
    for inst in instancias:
        celdas = []
        for cid in ("GA-final", "GA-final-aleatoria"):
            f = [r for r in fin if r["config"] == cid and r["instancia"] == inst
                 and r["semilla"] in (100, 101, 102)]
            d = 0 if inst.startswith("X-") else 2
            vals = (media(r["costo_inicial"] for r in f), media(r["costo"] for r in f),
                    media(r["gap_bks_pct"] for r in f))
            celdas += [num(vals[0], d), num(vals[1], d), num(vals[2])]
            datos.setdefault(inst, {})[cid] = vals
        filas.append(" & ".join([inst, str(f[0]["n"])] + celdas) + r" \\")
    escribir("inicializacion.tex", [
        r"\begin{tabular}{lrrrrrrr}", r"\toprule",
        r" & & \multicolumn{3}{c}{Vecino más cercano aleatorizado} & "
        r"\multicolumn{3}{c}{Permutaciones aleatorias} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
        r"Instancia & $n$ & $Z$ inicial & $Z$ final & GAP (\%) & $Z$ inicial & $Z$ final & GAP (\%) \\",
        r"\midrule", *filas, r"\bottomrule", r"\end{tabular}"])
    resumen["inicializacion"] = datos


def tabla_resumen(fin, prop, matriz, resumen):
    filas = []
    ganadas = Counter()
    for k in range(len(INSTANCIAS)):
        mejor = min(matriz[m][k] for m in METODOS)
        ganadas.update(m for m in METODOS if matriz[m][k] - mejor < 1e-9)
    valores = {}
    for m in METODOS:
        if m in ("CW", "CW+BL"):
            filas_m = [prop[m, i] for i in INSTANCIAS]
            con_bks = sum(r["gap_bks_pct"] < TOL_BKS for r in filas_m)
            tiempo = media(r["tiempo"] for r in filas_m)
            t_mejor = None
            iteraciones = media(r["iteraciones"] for r in filas_m)
        else:
            filas_m = [r for r in fin if r["config"] == CONFIG_FINAL[m] and r["instancia"] in INSTANCIAS]
            con_bks = len({r["instancia"] for r in filas_m if r["gap_bks_pct"] < TOL_BKS})
            tiempo = media(r["tiempo"] for r in filas_m)
            t_mejor = media(r["tiempo_hasta_mejor"] for r in filas_m)
            iteraciones = media(r["iteraciones"] for r in filas_m)
        v = matriz[m]
        valores[m] = [st.mean(v), st.mean(v[:14]), st.mean(v[14:]), max(v), ganadas[m], con_bks,
                      tiempo, t_mejor, iteraciones]
    columnas = list(zip(*valores.values()))
    formato = [negrita_min(list(columnas[0])), negrita_min(list(columnas[1])),
               negrita_min(list(columnas[2])), negrita_min(list(columnas[3])),
               negrita_min(list(columnas[4]), 0, True), negrita_min(list(columnas[5]), 0, True),
               [num(x, 3) if x < 1 else num(x, 1) for x in columnas[6]],
               [num(x, 1) for x in columnas[7]], [num(x, 0) for x in columnas[8]]]
    for k, m in enumerate(METODOS):
        filas.append(" & ".join([NOMBRE[m]] + [col[k] for col in formato]) + r" \\")
    escribir("resumen.tex", [
        r"\begin{tabular}{lrrrrrrrrr}", r"\toprule",
        r"Método & \makecell{GAP\\prom. (\%)} & \makecell{GAP\\CMT (\%)} & \makecell{GAP\\X (\%)} & "
        r"\makecell{GAP\\máx. (\%)} & \makecell{Inst.\\ganadas} & \makecell{Inst.\\con BKS} & "
        r"\makecell{Tiempo\\prom. (s)} & \makecell{Tiempo al\\mejor (s)} & \makecell{Iteraciones\\prom.} \\",
        r"\midrule", *filas, r"\bottomrule", r"\end{tabular}"])
    resumen["resumen"] = {m: dict(zip(["gap", "gap_cmt", "gap_x", "gap_max", "ganadas", "con_bks",
                                        "tiempo", "t_mejor", "iteraciones"], v))
                          for m, v in valores.items()}
    resumen["bks_memetico"] = dict(Counter(r["instancia"] for r in fin
                                           if r["config"] == "GA-memetico"
                                           and r["instancia"] in INSTANCIAS
                                           and r["gap_bks_pct"] < TOL_BKS))


def resumen_rutas(fin, prop, resumen):
    datos = {}
    for inst in INSTANCIAS_X:
        datos[inst] = {m: sorted({r["numero_rutas"] for r in fin
                                  if r["config"] == CONFIG_FINAL[m] and r["instancia"] == inst})
                       for m in ("GA", "ACO", "MEM")}
        for m in ("CW", "CW+BL"):
            datos[inst][m] = [prop[m, inst]["numero_rutas"]]
    resumen["rutas_x"] = datos


# --- figuras ---

def guardar(fig, nombre):
    FIGURAS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURAS / nombre)
    plt.close(fig)


def figura_exacto(exa, gap_exacto):
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6))
    color = {"optimo": "#1aa37a", "factible": "#f0a30a", "sin_solucion": "#e8643a"}
    etiqueta = {"optimo": "óptimo demostrado", "factible": "factible (600 s)",
                "sin_solucion": "sin solución (600 s)"}
    for est in color:
        pts = [(r["n"], r["tiempo"]) for r in exa.values() if r["estatus"] == est]
        a.scatter(*zip(*pts), color=color[est], label=etiqueta[est], zorder=3)
    vistos = Counter()
    for nombre, r in exa.items():
        en_limite = r["tiempo"] > 500
        desplazamiento = 6 + 22 * vistos[r["n"]] if en_limite else 3
        vistos[r["n"]] += en_limite
        a.annotate(nombre.replace("CMT1-", ""), (r["n"], r["tiempo"]), fontsize=6,
                   xytext=(2, desplazamiento), textcoords="offset points",
                   rotation=60 if en_limite else 0)
    a.axhline(600, ls=":", color="gray", lw=1)
    a.set_yscale("log")
    a.set_ylim(0.05, 2e4)
    a.set_xlabel("Clientes $n$")
    a.set_ylabel("Tiempo de CBC (s, escala log)")
    a.set_title("Tiempo del modelo exacto")
    a.legend(fontsize=7, loc="lower right")
    orden = list(gap_exacto)
    x = range(len(orden))
    for m in METODOS[1:]:
        b.plot(x, [gap_exacto[i][m] for i in orden], marker="o", ms=3, color=COLOR[m],
               label=NOMBRE_FIG[m])
    b.axhline(0, color="#5b4bd6", lw=1)
    b.axvspan(-0.3, 4.3, color="#1aa37a", alpha=0.08)
    b.text(2, b.get_ylim()[1] * 0.9 if b.get_ylim()[1] > 0 else 1, "óptimo demostrado",
           ha="center", fontsize=7, color="gray")
    b.set_xticks(list(x), [i.replace("CMT1-", "") for i in orden], rotation=45, fontsize=7)
    b.set_ylabel("GAP vs modelo exacto (%)")
    b.set_title("GAP de cada método respecto al exacto")
    b.legend(fontsize=7, loc="lower left")
    guardar(fig, "exacto.pdf")


def figura_efectos(efectos_ga, efectos_aco, mejores):
    paneles = [("GA", "puntos_cruce", "GA: puntos de cruce"), ("GA", "prob_mutacion", "GA: prob. mutación"),
               ("GA", "seleccion", "GA: selección"), ("ACO", "alfa", r"ACO: $\alpha$ (feromona)"),
               ("ACO", "beta", r"ACO: $\beta$ (visibilidad)"), ("ACO", "rho", r"ACO: $\rho$ (persistencia)")]
    fig, ejes = plt.subplots(1, 6, figsize=(11, 2.8), sharey=True)
    for eje, (alg, factor, titulo) in zip(ejes, paneles):
        efectos = efectos_ga if alg == "GA" else efectos_aco
        niveles = list(efectos[factor])
        valores = [efectos[factor][n] for n in niveles]
        colores = [COLOR[alg] if n == mejores[alg][factor] else "#cfcfcf" for n in niveles]
        barras = eje.bar([f"{n:g}" if isinstance(n, float) else str(n) for n in niveles], valores,
                         color=colores)
        for barra, v in zip(barras, valores):
            eje.text(barra.get_x() + barra.get_width() / 2, v + 0.1, f"{v:.2f}", ha="center", fontsize=7)
        eje.set_title(titulo, fontsize=8)
    ejes[0].set_ylabel("Rango promedio (menor = mejor)")
    guardar(fig, "efectos.pdf")


def figura_factorial(tablas):
    fig, eje = plt.subplots(figsize=(7.5, 4))
    for alg, marcador in (("GA", "o"), ("ACO", "s")):
        for cid, t in tablas[alg].items():
            eje.scatter(t["t_mejor"], t["gap"], color=COLOR[alg], marker=marcador, s=30, zorder=3)
            eje.annotate(cid.replace("GA-", "").replace("ACO-", ""), (t["t_mejor"], t["gap"]),
                         fontsize=6, xytext=(3, 2), textcoords="offset points")
        eje.scatter([], [], color=COLOR[alg], marker=marcador, label=alg)
    eje.set_xlabel("Tiempo promedio hasta la mejor solución (s)")
    eje.set_ylabel("GAP promedio vs BKS (%)")
    eje.set_title("Factorial: calidad contra tiempo (24 instancias, 3 semillas)")
    eje.legend()
    guardar(fig, "factorial.pdf")


def figura_gap_bks(matriz):
    orden = sorted(range(24), key=lambda k: (k >= 14, cargar(INSTANCIAS[k], DATOS)["n"], k))
    fig, ejes = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw={"width_ratios": [14, 10]})
    for eje, grupo, titulo in ((ejes[0], [k for k in orden if k < 14], "Serie CMT"),
                               (ejes[1], [k for k in orden if k >= 14], "Serie X")):
        ancho = 0.16
        for j, m in enumerate(METODOS):
            eje.bar([p + (j - 2) * ancho for p in range(len(grupo))], [matriz[m][k] for k in grupo],
                    ancho, color=COLOR[m], label=NOMBRE_FIG[m])
        eje.set_xticks(range(len(grupo)), [corto(INSTANCIAS[k]) for k in grupo], rotation=45,
                       fontsize=7, ha="right")
        eje.set_title(titulo)
    ejes[0].set_ylabel("GAP promedio vs BKS (%)")
    ejes[1].legend(fontsize=7)
    guardar(fig, "gap_bks.pdf")


def _curva(trazas, bks, rejilla):
    """Mediana del GAP de la mejor solución en cada instante de la rejilla."""
    curvas = []
    for traza in trazas:
        valores, k, actual = [], 0, None
        for t in rejilla:
            while k < len(traza) and traza[k][1] <= t:
                actual = traza[k][2]
                k += 1
            valores.append(None if actual is None else (actual - bks) / bks * 100)
        curvas.append(valores)
    return [st.median([c[j] for c in curvas if c[j] is not None])
            if any(c[j] is not None for c in curvas) else None for j in range(len(rejilla))]


def figura_convergencia(fin, prop):
    fig, ejes = plt.subplots(1, 2, figsize=(10, 3.4))
    for eje, inst in zip(ejes, ("CMT5", "X-n459-k26")):
        bks = prop["CW", inst]["bks"]
        tmax = tiempo_max(prop["CW", inst]["n"])
        rejilla = [tmax * j / 400 for j in range(401)]
        for m in ("GA", "ACO", "MEM"):
            trazas = [r["traza"] for r in fin if r["config"] == CONFIG_FINAL[m] and r["instancia"] == inst]
            curva = _curva(trazas, bks, rejilla)
            eje.plot(rejilla, curva, color=COLOR[m], label=NOMBRE_FIG[m], drawstyle="steps-post")
        eje.axhline(prop["CW+BL", inst]["gap_bks_pct"], color=COLOR["CW+BL"], ls="--",
                    label=NOMBRE_FIG["CW+BL"])
        eje.axhline(0, color="black", ls=":", lw=1, label="BKS")
        eje.set_ylim(-1, 45)
        eje.set_xlabel("Tiempo (s)")
        eje.set_ylabel("GAP de la mejor solución (%)")
        eje.set_title(f"{inst} (mediana de 5 semillas)")
    ejes[0].legend(fontsize=7)
    guardar(fig, "convergencia.pdf")


def figura_rutas(fin, inst, nombre_archivo):
    filas = [r for r in fin if r["config"] == "GA-memetico" and r["instancia"] == inst]
    mejor = min(filas, key=lambda r: (r["costo"], r["semilla"]))
    instancia = cargar(inst, DATOS)
    coords = instancia["coordenadas"]
    fig, eje = plt.subplots(figsize=(5.2, 5))
    colores = plt.cm.tab10.colors
    for k, ruta in enumerate(mejor["rutas"]):
        xs = [coords[0][0]] + [coords[c][0] for c in ruta] + [coords[0][0]]
        ys = [coords[0][1]] + [coords[c][1] for c in ruta] + [coords[0][1]]
        carga = sum(instancia["demandas"][c] for c in ruta)
        eje.plot(xs, ys, marker="o", ms=3.5, lw=1.2, color=colores[k % 10],
                 label=f"Ruta {k + 1}: {len(ruta)} clientes, carga {carga}/{instancia['capacidad']}")
    eje.scatter(*coords[0], marker="s", s=80, color="#d62728", zorder=5, label="Depósito")
    eje.set_title(f"{inst}, GA memético (semilla {mejor['semilla']}): Z = {mejor['costo']:.2f}")
    eje.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=2)
    eje.set_aspect("equal")
    guardar(fig, nombre_archivo)
    return mejor


def main():
    registros, fac, fin, prop, exa = cargar_datos()
    resumen = {"corridas": len(registros),
               "soluciones": sum(1 for r in registros if r.get("rutas")),
               "validas": sum(1 for r in registros if r.get("valida") is True)}
    tabla_instancias()
    tabla_exacto(exa, resumen)
    gap_exacto = tabla_gap_exacto(exa, fin, prop, resumen)
    tablas, efectos, mejores = {}, {}, {}
    for alg in ("GA", "ACO"):
        mejores[alg], efectos[alg], tablas[alg] = resumen_factorial(fac, alg)
        tabla_factorial(alg, tablas[alg], f"factorial_{alg.lower()}.tex")
        tabla_efectos(alg, tablas[alg], efectos[alg], mejores[alg], f"efectos_{alg.lower()}.tex")
    resumen["mejores"] = mejores
    resumen["factorial"] = tablas
    tabla_confirmacion(fac, fin, tablas, mejores, resumen)
    matriz = tabla_gap_bks(fin, prop, resumen)
    tabla_wilcoxon(matriz, resumen)
    tabla_tiempos(fin, prop, resumen)
    tabla_inicializacion(fin, resumen)
    tabla_resumen(fin, prop, matriz, resumen)
    resumen_rutas(fin, prop, resumen)
    figura_exacto(exa, gap_exacto)
    figura_efectos(efectos["GA"], efectos["ACO"], mejores)
    figura_factorial(tablas)
    figura_gap_bks(matriz)
    figura_convergencia(fin, prop)
    mejor_cmt1 = figura_rutas(fin, "CMT1", "rutas_cmt1.pdf")
    resumen["figura_rutas_cmt1"] = {"semilla": mejor_cmt1["semilla"], "costo": mejor_cmt1["costo"]}
    with open(ETAPA / "resumen.json", "w", encoding="utf-8") as archivo:
        json.dump(resumen, archivo, indent=1, ensure_ascii=False, default=str)
    print(f"Tablas en {TABLAS}")
    print(f"Figuras en {FIGURAS}")
    print(f"Cifras del texto en {ETAPA / 'resumen.json'}")


if __name__ == "__main__":
    main()

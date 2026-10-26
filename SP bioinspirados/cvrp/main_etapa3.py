"""Etapa 3: experimentos sobre las 24 instancias de CVRPLIB (CMT1-14 y 10 instancias X).

Fases (se pueden correr por separado o todas con --fase todo):
  propuesto  Clarke-Wright y Clarke-Wright + búsqueda local (deterministas).
  factorial  GA: 2 x 3 x 2 = 12 combinaciones; ACO: 2 x 2 x 2 = 8; 3 semillas por caso.
  exacto     Modelo MTZ con límite de tiempo en subinstancias de CMT1 y en las CMT sin
             duración máxima, para ver hasta dónde escala.
  final      Mejores niveles de cada factor (elegidos con el factorial), 5 semillas nuevas,
             más la versión memética del GA y la comparación de la población inicial.

Uso:
  python main_etapa3.py --prueba            # corrida corta para revisar que todo funcione
  python main_etapa3.py --fase todo         # experimento completo (varias horas)
  python main_etapa3.py --fase factorial --procesos 6

Si se interrumpe (Ctrl+C), al volver a ejecutar el mismo comando continúa donde se quedó.
"""

import argparse
import json
import multiprocessing as mp
import platform
import sys
from pathlib import Path

import pulp

from src import experimentos3 as ex
from src.instancias_cvrplib import INSTANCIAS

RAIZ = Path(__file__).resolve().parent
DATOS = RAIZ / "datos"


def titulo(texto):
    print(f"\n{'=' * 78}\n{texto}\n{'=' * 78}", flush=True)


def nombre_cpu():
    try:
        if sys.platform == "win32":
            import winreg
            llave = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                   r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(llave, "ProcessorNameString")[0].strip()
        if sys.platform == "darwin":
            import subprocess
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True, check=True).stdout.strip()
        with open("/proc/cpuinfo", encoding="utf-8") as archivo:
            for linea in archivo:
                if linea.startswith("model name"):
                    return linea.split(":", 1)[1].strip()
    except (OSError, ImportError, Exception):
        pass
    return platform.processor() or "desconocido"


def configuracion(args):
    if args.prueba:
        return {
            "salida": RAIZ / "resultados" / "etapa3_prueba",
            "instancias": ["CMT1", "CMT6", "X-n280-k17"],
            "subinstancias": ["CMT1-k10", "CMT1-k15"],
            "exacto": ["CMT1-k10", "CMT1-k15", "CMT1"],
            "limite_exacto": 20,
            "escala": 0.1,
            "semillas_factorial": [0],
            "semillas_final": [100],
            "inicializacion": ["CMT1"],
            "semillas_inicializacion": [100],
        }
    instancias = args.instancias or INSTANCIAS
    return {
        "salida": RAIZ / "resultados" / "etapa3",
        "instancias": instancias,
        "subinstancias": ex.SUBINSTANCIAS,
        "exacto": ex.SUBINSTANCIAS + ex.EXACTO_COMPLETAS,
        "limite_exacto": args.limite_exacto,
        "escala": args.escala,
        "semillas_factorial": ex.SEMILLAS_FACTORIAL,
        "semillas_final": ex.SEMILLAS_FINAL,
        "inicializacion": ex.INSTANCIAS_INICIALIZACION,
        "semillas_inicializacion": ex.SEMILLAS_FINAL[:3],
    }


def imprimir_efectos(algoritmo, mejores, efectos, rango_promedio, instancias):
    print(f"{algoritmo}: rango promedio de cada nivel en {len(instancias)} instancias "
          f"(1 = mejor configuración de la instancia)")
    for factor, valores in efectos.items():
        texto = "   ".join(f"{nivel}: {rango:.2f}" for nivel, rango in valores.items())
        print(f"  {factor:<14} {texto}   -> mejor: {mejores[factor]}")
    mejor_combinacion = min(rango_promedio, key=rango_promedio.get)
    print(f"  Mejor combinación del factorial: {mejor_combinacion} "
          f"(rango promedio {rango_promedio[mejor_combinacion]:.2f})")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fase", default="todo",
                        choices=["propuesto", "factorial", "exacto", "final", "todo"])
    parser.add_argument("--procesos", type=int, default=max(1, (mp.cpu_count() or 2) // 2),
                        help="procesos en paralelo (por defecto, la mitad de los hilos)")
    parser.add_argument("--escala", type=float, default=1.0,
                        help="multiplica el presupuesto de tiempo de GA y ACO")
    parser.add_argument("--limite-exacto", type=int, default=ex.LIMITE_EXACTO,
                        help="límite de tiempo del modelo exacto por instancia (s)")
    parser.add_argument("--instancias", nargs="*", help="subconjunto de instancias")
    parser.add_argument("--prueba", action="store_true", help="corrida corta de verificación")
    args = parser.parse_args()

    cfg = configuracion(args)
    salida = cfg["salida"]
    salida.mkdir(parents=True, exist_ok=True)
    jsonl = salida / "corridas.jsonl"
    carpeta = str(DATOS)

    equipo = {"cpu": nombre_cpu(), "sistema": platform.platform(),
              "python": platform.python_version(), "pulp": pulp.__version__,
              "procesos": args.procesos, "escala": cfg["escala"]}
    with open(salida / "equipo.json", "w", encoding="utf-8") as archivo:
        json.dump(equipo, archivo, indent=2, ensure_ascii=False)
    titulo("ETAPA 3: CONFIGURACIÓN")
    for k, v in equipo.items():
        print(f"{k:<9} {v}")
    print(f"Instancias: {len(cfg['instancias'])}   salida: {salida}")

    fases = ["propuesto", "factorial", "exacto", "final"] if args.fase == "todo" else [args.fase]

    if "propuesto" in fases:
        titulo("FASE propuesto: Clarke-Wright y Clarke-Wright + búsqueda local")
        ex.ejecutar_tareas(ex.tareas_propuesto(carpeta, cfg["instancias"] + cfg["subinstancias"]),
                           jsonl, args.procesos)

    if "factorial" in fases:
        titulo("FASE factorial: GA (12 combinaciones) y ACO (8 combinaciones)")
        ex.ejecutar_tareas(ex.tareas_factorial(carpeta, cfg["instancias"],
                                               cfg["semillas_factorial"], cfg["escala"]),
                           jsonl, args.procesos)

    if "exacto" in fases:
        titulo(f"FASE exacto: modelo MTZ con límite de {cfg['limite_exacto']} s")
        ex.ejecutar_tareas(ex.tareas_exacto(carpeta, cfg["exacto"], cfg["limite_exacto"],
                                            salida / "logs_cbc"), jsonl, args.procesos)

    if "final" in fases:
        titulo("FASE final: mejores niveles de cada hiperparámetro")
        registros = ex.leer_corridas(jsonl)
        mejores = {}
        for algoritmo in ("GA", "ACO"):
            niveles, efectos, rango_promedio, instancias = ex.seleccionar_mejores(
                registros, algoritmo)
            mejores[algoritmo] = niveles
            imprimir_efectos(algoritmo, niveles, efectos, rango_promedio, instancias)
        with open(salida / "mejores_parametros.json", "w", encoding="utf-8") as archivo:
            json.dump(mejores, archivo, indent=2, ensure_ascii=False)
        ex.ejecutar_tareas(ex.tareas_final(carpeta, cfg["instancias"] + cfg["subinstancias"],
                                           cfg["semillas_final"], cfg["escala"], mejores,
                                           cfg["inicializacion"],
                                           cfg["semillas_inicializacion"]),
                           jsonl, args.procesos)

    ex.exportar_csv(jsonl, salida / "corridas.csv")
    titulo("ARCHIVOS")
    print(jsonl)
    print(salida / "corridas.csv")
    print(salida / "equipo.json")


if __name__ == "__main__":
    mp.freeze_support()
    main()
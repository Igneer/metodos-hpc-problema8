#!/usr/bin/env python3
"""Metricas a partir de crudo.csv (salida de correr_benchmarks.sh).

Uso:
  python scripts/metricas.py                                  # resultados/servidor/crudo.csv
  python scripts/metricas.py --entradas A/crudo.csv B/crudo.csv   # varias corridas: se concatenan
  python scripts/metricas.py --salida resultados/servidor

Escribe en <salida>/metricas.csv. Requiere pandas y numpy.

Definiciones (todo sobre la MEDIANA de las repeticiones de cada (algoritmo, n, p)):
  mad_s / mad_rel      mediana de |x - mediana| y su cociente con la mediana.
  speedup_p1           t(alg, n, p=1) / t(alg, n, p): escalado del propio algoritmo.
  speedup_mejor_serial t(mejor serial para ese n) / t(alg, n, p).
  eficiencia_*         speedup / p.
  fraccion_serial_karp_flatt           fraccion serial experimental e = (1/S - 1/p) / (1 - 1/p), con S = speedup_p1.
  sobrecosto_paralelo_s  t(p) - t(p=1)/p: tiempo perdido respecto del escalado ideal del propio algoritmo
                         (comunicacion, contencion de memoria, desbalance; vacio para p=1 y seriales).
  sobrecosto_por_n2_ns   sobrecosto_paralelo_s / n^2, en nanosegundos: si es constante el sobrecosto
                         escala como n^2 (p. ej. el Bcast de B); si crece, hay saturacion (memoria/ancho de banda).
  fraccion_generacion             gen / (gen + t): peso de la generacion (no paralelizable) en la corrida completa.
  razon_generacion_ejecucion          gen / t.
  speedup_con_generacion_mejor_serial  (gen + t_mejor_serial) / (gen + t): speedup de la corrida completa
                         incluyendo la generacion secuencial (Amdahl real).
  tiempo_relativo_pdgemm            t / t(ref_pdgemm, mismo n y p): >1 = mas lento que ScaLAPACK.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SERIALES = ["serial_ikj", "serial_ikj_bloqueado"]


def mad(x):
    x = np.asarray(x, float)
    return float(np.median(np.abs(x - np.median(x))))


def cargar(entradas):
    d = pd.concat([pd.read_csv(f) for f in entradas], ignore_index=True)
    # misma medicion en varias corridas: gana la ultima entrada
    return d.drop_duplicates(["algoritmo", "n", "procesos", "repeticion"], keep="last")


def calcular(d):
    filas = []
    for (alg, n, p), g in d.groupby(["algoritmo", "n", "procesos"]):
        t = g.segundos.to_numpy()
        med, m = float(np.median(t)), mad(t)
        gen = float(g.segundos_generacion.median())
        filas.append(dict(algoritmo=alg, n=n, procesos=p, reps=len(t),
                          mediana_s=med, mad_s=m, mad_rel=m / med, min_s=t.min(), max_s=t.max(),
                          gflops=float(g.gflops.median()), generacion_s=gen,
                          fraccion_generacion=gen / (gen + med), razon_generacion_ejecucion=gen / med))
    r = pd.DataFrame(filas)
    med = r.set_index(["algoritmo", "n", "procesos"]).mediana_s

    def base(alg, n, p):
        return med.get((alg, n, p), np.nan)

    mejor = {}
    for n, g in r[r.algoritmo.isin(SERIALES)].groupby("n"):
        mejor[n] = g.mediana_s.min()
    r["serial_mejor_s"] = r.n.map(mejor)
    r["p1_s"] = [base(a, n, 1) for a, n in zip(r.algoritmo, r.n)]
    r["t_ref_s"] = [base("ref_pdgemm", n, p) for n, p in zip(r.n, r.procesos)]

    mpi = ~r.algoritmo.isin(SERIALES)
    for nombre, b in [("p1", "p1_s"), ("mejor_serial", "serial_mejor_s")]:
        s = r[b] / r.mediana_s
        r[f"speedup_{nombre}"] = s.where(mpi)
        r[f"eficiencia_{nombre}"] = (s / r.procesos).where(mpi)
    S, p = r.speedup_p1, r.procesos
    r["fraccion_serial_karp_flatt"] = ((1 / S - 1 / p) / (1 - 1 / p)).where(mpi & (p > 1))
    r["sobrecosto_paralelo_s"] = (r.mediana_s - r.p1_s / p).where(mpi & (p > 1))
    r["sobrecosto_por_n2_ns"] = r.sobrecosto_paralelo_s / r.n ** 2 * 1e9
    r["speedup_con_generacion_mejor_serial"] = ((r.generacion_s + r.serial_mejor_s) / (r.generacion_s + r.mediana_s)).where(mpi)
    r["tiempo_relativo_pdgemm"] = r.mediana_s / r.t_ref_s
    return r.drop(columns=["p1_s", "t_ref_s"]).sort_values(["algoritmo", "n", "procesos"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entradas", nargs="+", default=["resultados/servidor/crudo.csv"])
    ap.add_argument("--salida", default="resultados/servidor")
    a = ap.parse_args()
    d = cargar(a.entradas)
    r = calcular(d)
    salida = Path(a.salida)
    salida.mkdir(parents=True, exist_ok=True)
    r.to_csv(salida / "metricas.csv", index=False, float_format="%.6g")
    print(f"{len(d)} mediciones -> {len(r)} filas en {salida/'metricas.csv'}")


if __name__ == "__main__":
    main()

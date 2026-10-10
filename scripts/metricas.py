#!/usr/bin/env python3

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
                          fraccion_generacion=gen / (gen + med)))
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
        if nombre == "p1":
            r["eficiencia_p1"] = (s / r.procesos).where(mpi)
    S, p = r.speedup_p1, r.procesos
    r["fraccion_serial_karp_flatt"] = ((1 / S - 1 / p) / (1 - 1 / p)).where(mpi & (p > 1))
    r["sobrecosto_paralelo_s"] = (r.mediana_s - r.p1_s / p).where(mpi & (p > 1))
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

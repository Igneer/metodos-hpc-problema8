#!/usr/bin/env python3
"""Compara el i5-7400 (4 cores) con el EPYC (48 cores) para los p comunes (1, 2, 4).

Uso:
  python scripts/comparar_equipos.py
  python scripts/comparar_equipos.py --n 10000 --salida img/equipos_i5_vs_epyc.png

Lee los metricas.csv de cada equipo (medianas). Eficiencia relativa al propio p = 1.
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

TINTA = "#2b2b2b"
ALGS = [("mpi_ikj", "ikj 1D", "#2a78d6"), ("mpi_ikj_bloqueado", "ikj blocking 1D", "#eb6834"),
        ("ref_pdgemm", "PDGEMM 2D", "#2f9e6b")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--i5", default="resultados/server_i5_7400_en_ram/metricas.csv")
    ap.add_argument("--epyc", default="resultados/servidor/metricas.csv")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--salida", default="img/equipos_i5_vs_epyc.png")
    a = ap.parse_args()

    eq = [("i5-7400 (4 cores)", pd.read_csv(a.i5), "-", "o"), ("EPYC (48 cores)", pd.read_csv(a.epyc), "--", "s")]
    fig, ejes = plt.subplots(1, 2, figsize=(11, 4.4))
    for nombre, d, ls, mk in eq:
        d = d[(d.n == a.n) & (d.procesos <= 4)]
        for alg, etiqueta, color in ALGS:
            s = d[d.algoritmo == alg].sort_values("procesos")
            ejes[0].plot(s.procesos, s.gflops, ls, marker=mk, color=color, label=f"{etiqueta} · {nombre}")
            ejes[1].plot(s.procesos, s.eficiencia_p1, ls, marker=mk, color=color)
    ejes[0].set_ylabel("GFLOPS")
    ejes[0].set_title(f"Rendimiento, n = {a.n}", loc="left", fontsize=11, color=TINTA)
    ejes[1].set_ylabel("Eficiencia $E = S_{rel}/p$")
    ejes[1].set_title(f"Eficiencia, n = {a.n}", loc="left", fontsize=11, color=TINTA)
    ejes[1].set_ylim(0, 1.1)
    for ax in ejes:
        ax.set_xlabel("p")
        ax.set_xticks([1, 2, 4])
        ax.grid(alpha=0.25)
    ejes[0].legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(a.salida, dpi=150)
    print(f"Grafico en {a.salida}")


if __name__ == "__main__":
    main()

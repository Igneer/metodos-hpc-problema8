#!/usr/bin/env python3
"""Compara las versiones bloqueadas medidas con b=64 y con b=160.

Uso:
  python scripts/comparar_bloques.py
  python scripts/comparar_bloques.py --b64 resultados/servidor/crudo.csv \
      --b160 resultados/servidor/crudo_mayor_bloque.csv --salida informe/img/bloque_64_vs_160.png

Todo sobre la mediana de las repeticiones. S_abs = speedup contra el serial bloqueado
del mismo b. Solo se comparan los p presentes en ambas corridas.
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

C64, C160, TINTA = "#2a78d6", "#eb6834", "#2b2b2b"


def cargar(ruta):
    d = pd.read_csv(ruta, dtype=str)
    d = d[d.perfil != "perfil"]  # por si quedo una cabecera repetida
    for c in ["n", "procesos", "repeticion", "segundos", "gflops"]:
        d[c] = pd.to_numeric(d[c])
    d = d[d.algoritmo.str.contains("bloqueado")]
    m = d.groupby(["algoritmo", "n", "procesos"])[["segundos", "gflops"]].median().reset_index()
    return m.rename(columns={"segundos": "t", "gflops": "g"})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--b64", default="resultados/servidor/crudo.csv")
    ap.add_argument("--b160", default="resultados/servidor/crudo_mayor_bloque.csv")
    ap.add_argument("--salida", default="informe/img/bloque_64_vs_160.png")
    ap.add_argument("--p", type=int, default=48, help="p de los paneles paralelos")
    a = ap.parse_args()

    m = cargar(a.b64).merge(cargar(a.b160), on=["algoritmo", "n", "procesos"], suffixes=("64", "160"))
    ser = m[m.algoritmo == "serial_ikj_bloqueado"].set_index("n")
    par = m[(m.algoritmo == "mpi_ikj_bloqueado") & (m.procesos == a.p)].set_index("n")
    par = par.assign(S64=ser.t64 / par.t64, S160=ser.t160 / par.t160)

    fig, ejes = plt.subplots(1, 3, figsize=(15, 4.4))
    paneles = [
        (ejes[0], ser, "g", "GFLOPS", "Serial bloqueado (p = 1)"),
        (ejes[1], par, "g", "GFLOPS", f"1D bloqueada, p = {a.p}"),
        (ejes[2], par, "S", "Speedup absoluto $S_{abs}$", f"1D bloqueada, p = {a.p}: speedup"),
    ]
    for ax, d, col, ylab, titulo in paneles:
        ax.plot(d.index, d[f"{col}64"], "o-", color=C64, label="b = 64")
        ax.plot(d.index, d[f"{col}160"], "s-", color=C160, label="b = 160")
        ax.set_xlabel("n")
        ax.set_ylabel(ylab)
        ax.set_title(titulo, loc="left", fontsize=11, color=TINTA)
        ax.grid(alpha=0.25)
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False)
    ejes[2].plot(par.index, [a.p] * len(par), ":", color="gray", lw=1)
    ejes[2].text(par.index[0], a.p, " ideal", color="gray", va="bottom", fontsize=9)
    ejes[2].set_ylim(0, a.p * 1.1)
    fig.suptitle("Bloques de 64 y de 160 (medianas)", x=0.01, ha="left", fontsize=13, color=TINTA)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(a.salida, dpi=150)
    print(f"Grafico en {a.salida}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Metricas y graficas a partir de crudo.csv (salida de correr_benchmarks.sh).

Uso:
  python scripts/metricas.py                                  # resultados/servidor/crudo.csv
  python scripts/metricas.py --entradas A/crudo.csv B/crudo.csv   # varias corridas: se concatenan
  python scripts/metricas.py --salida resultados/servidor

Escribe en <salida>/: metricas.csv y graficas/*.png. Requiere pandas, numpy, matplotlib.

Definiciones (todo sobre la MEDIANA de las repeticiones de cada (algoritmo, n, p)):
  mad_s / mad_rel      mediana de |x - mediana| y su cociente con la mediana.
  speedup_p1           t(alg, n, p=1) / t(alg, n, p): escalado del propio algoritmo.
  speedup_serial_propio  t(serial equivalente) / t(alg, n, p). Equivalentes:
                         mpi_ikj -> serial_ikj, mpi_ikj_bloqueado -> serial_ikj_bloqueado
                         (ref_pdgemm no tiene serial propio: queda vacio).
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
  repeticiones_atipicas           repeticiones a mas de 3 MAD escaladas (1.4826*MAD) de la mediana.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SERIAL_PROPIO = {"mpi_ikj": "serial_ikj", "mpi_ikj_bloqueado": "serial_ikj_bloqueado"}
SERIALES = ["serial_ikj", "serial_ikj_bloqueado"]
# Slots categoricos 1,2,3,5 de la paleta de referencia (azul, naranja, agua, magenta) + marcador
# distinto por algoritmo, para que la identidad no dependa solo del color.
ESTILO = {
    "mpi_ikj":              ("#2a78d6", "o"),
    "mpi_ikj_bloqueado":    ("#eb6834", "s"),
    "ref_pdgemm":           ("#1baf7a", "^"),
    "serial_ikj_bloqueado": ("#e87ba4", "D"),
    "serial_ikj":           ("#8a8a85", "v"),
}
TINTA, TINTA2, GRILLA = "#1a1a19", "#5f5e58", "#e3e2dc"


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
        esc = 1.4826 * m
        atip = int((np.abs(t - med) > 3 * esc).sum()) if esc > 0 else 0
        gen = float(g.segundos_generacion.median())
        filas.append(dict(algoritmo=alg, n=n, procesos=p, reps=len(t),
                          mediana_s=med, mad_s=m, mad_rel=m / med, min_s=t.min(), max_s=t.max(),
                          repeticiones_atipicas=atip, gflops=float(g.gflops.median()), generacion_s=gen,
                          fraccion_generacion=gen / (gen + med), razon_generacion_ejecucion=gen / med))
    r = pd.DataFrame(filas)
    med = r.set_index(["algoritmo", "n", "procesos"]).mediana_s

    def base(alg, n, p):
        return med.get((alg, n, p), np.nan)

    mejor = {}
    for n, g in r[r.algoritmo.isin(SERIALES)].groupby("n"):
        mejor[n] = g.mediana_s.min()
    r["serial_mejor_s"] = r.n.map(mejor)
    r["serial_propio_s"] = [base(SERIAL_PROPIO.get(a), n, 1) if a in SERIAL_PROPIO else np.nan
                            for a, n in zip(r.algoritmo, r.n)]
    r["p1_s"] = [base(a, n, 1) for a, n in zip(r.algoritmo, r.n)]
    r["t_ref_s"] = [base("ref_pdgemm", n, p) for n, p in zip(r.n, r.procesos)]

    mpi = ~r.algoritmo.isin(SERIALES)
    for nombre, b in [("p1", "p1_s"), ("serial_propio", "serial_propio_s"),
                      ("mejor_serial", "serial_mejor_s")]:
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


# ---------------------------------------------------------------- graficas
def _estilo_ejes(ax):
    ax.set_facecolor("white")
    ax.grid(True, color=GRILLA, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRILLA)
    ax.tick_params(colors=TINTA2, labelsize=9)


# N de las facetas (un panel por N). Los que no esten en los datos se omiten.
NS_FACETAS = [500, 1000, 3000, 5000, 7500, 10000]


def _tamanios(r):
    ns = sorted(r.n.unique())
    elegidos = [n for n in NS_FACETAS if n in ns]
    return elegidos or sorted({ns[0], ns[len(ns) // 2], ns[-1]})


def _facetas(r, ns, ycol, ylabel, titulo, archivo, algs, ideal=None, logy=False, yerr=None,
             hline=None, xcol="procesos", xlabel="procesos (p)"):
    fig, axs = plt.subplots(1, len(ns), figsize=(4.6 * len(ns), 4.2), sharey=False, squeeze=False)
    fig.patch.set_facecolor("white")
    for ax, n in zip(axs[0], ns):
        _estilo_ejes(ax)
        for alg in algs:
            g = r[(r.algoritmo == alg) & (r.n == n)].dropna(subset=[ycol]).sort_values(xcol)
            if g.empty:
                continue
            c, mk = ESTILO[alg]
            if yerr and yerr in g:
                ax.errorbar(g[xcol], g[ycol], yerr=g[yerr], color=c, lw=2, marker=mk, ms=6,
                            mec="white", mew=1, capsize=2, label=alg)
            else:
                ax.plot(g[xcol], g[ycol], color=c, lw=2, marker=mk, ms=6, mec="white", mew=1, label=alg)
        if ideal == "lineal":
            ps = np.array(sorted(r.procesos.unique()))
            ax.plot(ps, ps, color=TINTA2, lw=1, ls="--", label="ideal")
        if hline is not None:
            ax.axhline(hline, color=TINTA2, lw=1, ls="--")
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(r.procesos.unique()))
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        if logy:
            ax.set_yscale("log")
        ax.set_title(f"n = {n}", color=TINTA, fontsize=11, loc="left")
        ax.set_xlabel(xlabel, color=TINTA2)
        ax.set_ylabel(ylabel, color=TINTA2)
    h, l = axs[0][-1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), frameon=False, labelcolor=TINTA2)
    fig.suptitle(titulo, color=TINTA, fontsize=13, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(archivo, dpi=150)
    plt.close(fig)


def graficar(r, carpeta):
    carpeta.mkdir(parents=True, exist_ok=True)
    ns = _tamanios(r)
    mpi = ["mpi_ikj", "mpi_ikj_bloqueado", "ref_pdgemm"]
    todos = [a for a in ESTILO if a in set(r.algoritmo)]

    _facetas(r, ns, "mediana_s", "tiempo (s), mediana ± MAD", "Tiempo de ejecución vs procesos",
             carpeta / "tiempo_vs_p.png", mpi, logy=True, yerr="mad_s")
    _facetas(r, ns, "speedup_mejor_serial", "speedup", "Speedup respecto del mejor serial",
             carpeta / "speedup_mejor_serial.png", mpi, ideal="lineal")
    _facetas(r, ns, "speedup_serial_propio", "speedup", "Speedup respecto de su propio serial",
             carpeta / "speedup_serial_propio.png", mpi, ideal="lineal")
    _facetas(r, ns, "speedup_p1", "speedup", "Escalado: speedup respecto de sí mismo con p=1",
             carpeta / "speedup_p1.png", mpi, ideal="lineal")
    _facetas(r, ns, "eficiencia_mejor_serial", "eficiencia", "Eficiencia paralela (respecto del mejor serial)",
             carpeta / "eficiencia_mejor_serial.png", mpi, hline=1.0)
    _facetas(r, ns, "eficiencia_serial_propio", "eficiencia", "Eficiencia paralela (respecto de su propio serial)",
             carpeta / "eficiencia_serial_propio.png", mpi, hline=1.0)
    _facetas(r, ns, "fraccion_serial_karp_flatt", "fracción serial experimental", "Karp-Flatt (sobre speedup_p1)",
             carpeta / "fraccion_serial_karp_flatt.png", mpi)
    _facetas(r, ns, "sobrecosto_paralelo_s", "sobrecosto t(p) − t(1)/p (s)",
             "Sobrecosto paralelo respecto del escalado ideal", carpeta / "sobrecosto_paralelo_vs_p.png",
             mpi, logy=True)
    _facetas(r, ns, "sobrecosto_por_n2_ns", "sobrecosto / n² (ns)",
             "Sobrecosto normalizado por n² (constante = escala como n²)",
             carpeta / "sobrecosto_por_n2_vs_p.png", mpi)
    _facetas(r, ns, "fraccion_generacion", "generación / (generación + t)", "Peso de la generación (parte no paralelizable)",
             carpeta / "generacion_fraccion_vs_p.png", mpi)
    _facetas(r, ns, "speedup_con_generacion_mejor_serial", "speedup",
             "Speedup de la corrida completa (incluye generación secuencial)",
             carpeta / "speedup_total_con_generacion.png", mpi, ideal="lineal")

    # tiempo y GFLOPS vs n para p=1 y p maximo
    pmax = int(r.procesos.max())
    for p, tag in [(1, "p1"), (pmax, f"p{pmax}")]:
        fig, axs = plt.subplots(1, 2, figsize=(9.5, 4.2))
        fig.patch.set_facecolor("white")
        for ax, col, yl in [(axs[0], "mediana_s", "tiempo (s), mediana ± MAD"), (axs[1], "gflops", "GFLOPS")]:
            _estilo_ejes(ax)
            for alg in todos:
                g = r[(r.algoritmo == alg) & (r.procesos == (1 if alg in SERIALES else p))].sort_values("n")
                if g.empty:
                    continue
                c, mk = ESTILO[alg]
                kw = dict(yerr=g.mad_s, capsize=2) if col == "mediana_s" else {}
                ax.errorbar(g.n, g[col], color=c, lw=2, marker=mk, ms=6, mec="white", mew=1, label=alg, **kw)
            ax.set_xscale("log")
            if col == "mediana_s":
                ax.set_yscale("log")
            ax.set_xlabel("n", color=TINTA2)
            ax.set_ylabel(yl, color=TINTA2)
        h, l = axs[1].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=len(l), frameon=False, labelcolor=TINTA2)
        fig.suptitle(f"Tiempo y GFLOPS vs n (MPI con p={p}; seriales con 1 proceso)", color=TINTA,
                     fontsize=13, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0.08, 1, 0.94))
        fig.savefig(carpeta / f"tiempo_gflops_vs_n_{tag}.png", dpi=150)
        plt.close(fig)

    # generacion: tiempo vs n (una curva, es independiente del algoritmo) y fraccion vs n por algoritmo
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 4.2))
    fig.patch.set_facecolor("white")
    g = r.groupby("n").generacion_s.median()
    _estilo_ejes(axs[0])
    axs[0].plot(g.index, g.values, color=ESTILO["mpi_ikj"][0], lw=2, marker="o", ms=6, mec="white", label="medida")
    n0 = g.index[0]
    axs[0].plot(g.index, g.values[0] * (g.index / n0) ** 2, color=TINTA2, lw=1, ls="--", label="∝ n²")
    axs[0].set_xscale("log"); axs[0].set_yscale("log")
    axs[0].set_xlabel("n", color=TINTA2); axs[0].set_ylabel("tiempo de generación (s)", color=TINTA2)
    axs[0].legend(frameon=False, labelcolor=TINTA2)
    _estilo_ejes(axs[1])
    for alg in todos:
        h = r[(r.algoritmo == alg) & (r.procesos == (1 if alg in SERIALES else pmax))].sort_values("n")
        if h.empty:
            continue
        c, mk = ESTILO[alg]
        axs[1].plot(h.n, h.fraccion_generacion, color=c, lw=2, marker=mk, ms=6, mec="white", label=alg)
    axs[1].set_xscale("log")
    axs[1].set_xlabel("n", color=TINTA2); axs[1].set_ylabel("generación / (generación + t)", color=TINTA2)
    axs[1].set_title(f"MPI con p={pmax}", color=TINTA, fontsize=10, loc="left")
    h, l = axs[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), frameon=False, labelcolor=TINTA2)
    fig.suptitle("Generación de las matrices (secuencial, solo rank 0)", color=TINTA, fontsize=13, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig.savefig(carpeta / "generacion_vs_n.png", dpi=150)
    plt.close(fig)

    # dispersion relativa (MAD/mediana) por n y p: mapa de calor por algoritmo
    fig, axs = plt.subplots(1, len(mpi), figsize=(4.8 * len(mpi), 4.2))
    for ax, alg in zip(axs, mpi):
        piv = r[r.algoritmo == alg].pivot(index="n", columns="procesos", values="mad_rel")
        im = ax.imshow(piv.values * 100, aspect="auto", cmap="Blues", origin="lower")
        ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns, fontsize=8)
        ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index, fontsize=8)
        ax.set_xlabel("procesos (p)", color=TINTA2); ax.set_ylabel("n", color=TINTA2)
        ax.set_title(alg, color=TINTA, fontsize=11, loc="left")
        fig.colorbar(im, ax=ax, label="MAD / mediana (%)")
    fig.suptitle("Dispersión relativa de las repeticiones", color=TINTA, fontsize=13, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(carpeta / "dispersion_relativa.png", dpi=150)
    plt.close(fig)


def main():
    global NS_FACETAS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entradas", nargs="+", default=["resultados/servidor/crudo.csv"])
    ap.add_argument("--salida", default="resultados/servidor")
    ap.add_argument("--ns", nargs="+", type=int, default=NS_FACETAS,
                    help="N de los paneles de las graficas (defecto: %(default)s)")
    a = ap.parse_args()
    NS_FACETAS = a.ns
    d = cargar(a.entradas)
    r = calcular(d)
    salida = Path(a.salida)
    salida.mkdir(parents=True, exist_ok=True)
    r.to_csv(salida / "metricas.csv", index=False, float_format="%.6g")
    graficar(r, salida / "graficas")
    print(f"{len(d)} mediciones -> {len(r)} filas en {salida/'metricas.csv'}; graficas en {salida/'graficas'}")
    if not (r.algoritmo == "serial_ikj").any() or r[r.algoritmo == "serial_ikj"].n.nunique() < r.n.nunique():
        print("AVISO: serial_ikj no esta para todos los n; speedup_serial_propio de mpi_ikj queda vacio donde falte.")
    for alg, g in r[r.repeticiones_atipicas > 0].groupby("algoritmo"):
        print(f"AVISO: {alg}: {int(g.repeticiones_atipicas.sum())} repeticiones atipicas en {len(g)} configuraciones")


if __name__ == "__main__":
    main()

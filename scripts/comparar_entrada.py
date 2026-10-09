#!/usr/bin/env python3
"""Compara la entrada por archivo (lectura de A y B) con la generacion en RAM.

Uso, desde la raiz del repositorio:
  python scripts/comparar_entrada.py

Lee los metricas.csv de las corridas en RAM y con archivo de cada equipo (EPYC e i5-7400),
escribe resultados/comparacion_entrada_archivo.csv y tres figuras en img/:
  entrada_lectura_vs_generacion.png   tiempo de entrada y fraccion de la corrida serial
  entrada_amdahl.png                  speedup de la corrida completa (entrada incluida)
  entrada_producto_archivo_vs_ram.png cociente de tiempos del producto, archivo / RAM
Los tiempos del producto no incluyen la entrada; segundos_generacion es la generacion
(corridas en RAM) o la lectura de los archivos con cache frio (corridas con archivo).
Requiere pandas, numpy y matplotlib.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R, OUT = "resultados/", "img/"
EQUIPOS = {  # nombre: (corrida en RAM, corrida con archivo, color, marcador)
    "EPYC (48 cores)": ("servidor", "servidor_48c_archivo", "#2a78d6", "o"),
    "i5-7400 (4 cores)": ("server_i5_7400_en_ram", "i5_7400_4c_archivo", "#eb6834", "s"),
}
BLOQ, IKJ, PDG = "mpi_ikj_bloqueado", "mpi_ikj", "ref_pdgemm"
ALGS = [(IKJ, "ikj 1D", "#2a78d6"), (BLOQ, "ikj blocking 1D", "#eb6834"), (PDG, "PDGEMM 2D", "#1baf7a")]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e3e2dc", "figure.dpi": 150,
                     "axes.edgecolor": "#5f5e58", "legend.frameon": False})
NS = [500, 1000, 2000, 5000, 10000]


def cargar(d):
    return pd.read_csv(f"{R}{d}/metricas.csv")


def entrada_por_n(m):
    """Tiempo de entrada (generacion o lectura) y serial bloqueado, por n."""
    s = m[m.algoritmo == "serial_ikj_bloqueado"].set_index("n")
    e = pd.DataFrame({"entrada_s": s.generacion_s, "serial_s": s.mediana_s})
    e["fraccion"] = e.entrada_s / (e.entrada_s + e.serial_s)
    e["cota_amdahl"] = 1 / e.fraccion
    return e


def tabla(datos):
    filas = []
    for eq, (ram, arc, _, _) in datos.items():
        a, b = entrada_por_n(cargar(ram)), entrada_por_n(cargar(arc))
        pmax = cargar(arc).procesos.max()
        mb = cargar(arc)
        mb = mb[(mb.algoritmo == BLOQ) & (mb.procesos == pmax)].set_index("n")
        mr = cargar(ram)
        mr = mr[(mr.algoritmo == BLOQ) & (mr.procesos == pmax)].set_index("n")
        for n in a.index:
            filas.append(dict(
                equipo=eq, n=n, generacion_ram_s=a.entrada_s[n], lectura_archivo_s=b.entrada_s[n],
                lectura_sobre_generacion=b.entrada_s[n] / a.entrada_s[n],
                lectura_mb_por_s=2 * n * n * 12.99 / 1e6 / b.entrada_s[n],  # ~12,99 B de texto por elemento (%.10g)
                fraccion_ram=a.fraccion[n], fraccion_archivo=b.fraccion[n],
                cota_amdahl_ram=a.cota_amdahl[n], cota_amdahl_archivo=b.cota_amdahl[n],
                lectura_min_s=cargar(arc).groupby('n').generacion_s.min()[n],
                lectura_mediana_s=cargar(arc).groupby('n').generacion_s.median()[n],
                lectura_max_s=cargar(arc).groupby('n').generacion_s.max()[n],
                p_max=pmax,
                speedup_total_ram=mr.speedup_con_generacion_mejor_serial[n],
                speedup_total_archivo=mb.speedup_con_generacion_mejor_serial[n],
                speedup_producto_ram=mr.speedup_mejor_serial[n],
                speedup_producto_archivo=mb.speedup_mejor_serial[n]))
    return pd.DataFrame(filas)


def fig_entrada(datos, t):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.9))
    for eq, (ram, arc, c, mk) in datos.items():
        a, b = entrada_por_n(cargar(ram)), entrada_por_n(cargar(arc))
        ax[0].plot(a.index, a.entrada_s, "--", marker=mk, color=c, mfc="none", label=f"generación en RAM · {eq}")
        ax[0].plot(b.index, b.entrada_s, "-", marker=mk, color=c, label=f"lectura de archivo · {eq}")
        ax[1].plot(a.index, 100 * a.fraccion, "--", marker=mk, color=c, mfc="none")
        ax[1].plot(b.index, 100 * b.fraccion, "-", marker=mk, color=c)
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("n"); ax[0].set_ylabel("tiempo de entrada (s)")
    ax[0].set_title("Tiempo de generar o leer A y B"); ax[0].legend(fontsize=7)
    ax[1].set_xscale("log"); ax[1].set_xlabel("n"); ax[1].set_ylabel("% de la corrida serial")
    ax[1].set_title("Peso de la entrada en la corrida serial completa")
    for a_ in ax:
        a_.set_xticks(NS); a_.set_xticklabels(NS); a_.minorticks_off()
    fig.tight_layout(); fig.savefig(OUT + "entrada_lectura_vs_generacion.png", bbox_inches="tight"); plt.close()


def fig_amdahl(datos):
    ram, arc, c, mk = datos["EPYC (48 cores)"]
    mr, ma = cargar(ram), cargar(arc)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.9))
    for alg, col in [(BLOQ, "#eb6834"), (PDG, "#1baf7a")]:
        etiqueta = dict((a, l) for a, l, _ in ALGS)[alg]
        for n, mkr in [(1000, "^"), (10000, "o")]:
            for m, ls, lab in [(mr, "--", "RAM"), (ma, "-", "archivo")]:
                d = m[(m.algoritmo == alg) & (m.n == n)].sort_values("procesos")
                ax[0].plot(d.procesos, d.speedup_con_generacion_mejor_serial, ls, marker=mkr, color=col, ms=4,
                           label=f"{etiqueta}, n={n} · {lab}")
    ax[0].set_xlabel("procesos p"); ax[0].set_ylabel("speedup de la corrida completa")
    ax[0].set_title("EPYC · entrada incluida (continuo: archivo, trazos: RAM)"); ax[0].legend(fontsize=6, ncol=2)
    for alg, col in [(BLOQ, "#eb6834"), (PDG, "#1baf7a")]:
        for m, ls, lab in [(mr, "--", "RAM"), (ma, "-", "archivo")]:
            d = m[(m.algoritmo == alg) & (m.procesos == 48)].sort_values("n")
            ax[1].plot(d.n, d.speedup_con_generacion_mejor_serial, ls, marker="s", ms=4, color=col,
                       label=f"{dict((a, l) for a, l, _ in ALGS)[alg]} · {lab}")
    for m, ls, lab in [(mr, "--", "RAM"), (ma, "-", "archivo")]:
        e = entrada_por_n(m)
        ax[1].plot(e.index, e.cota_amdahl, ls, color="#5f5e58", lw=1, label=f"cota de Amdahl 1/f · {lab}")
    ax[1].set_yscale("log"); ax[1].set_xscale("log")
    ax[1].set_xticks(NS); ax[1].set_xticklabels(NS); ax[1].minorticks_off()
    ax[1].set_yticks([1, 2, 5, 10, 20, 50]); ax[1].set_yticklabels([1, 2, 5, 10, 20, 50])
    ax[1].set_xlabel("n"); ax[1].set_ylabel("speedup"); ax[1].set_title("p = 48: medido y cota de Amdahl")
    ax[1].legend(fontsize=6.5)
    fig.tight_layout(); fig.savefig(OUT + "entrada_amdahl.png", bbox_inches="tight"); plt.close()


def fig_producto(datos):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.9), sharey=True)
    k = ["algoritmo", "n", "procesos"]
    for a, (eq, (ram, arc, c, mk)) in zip(ax, datos.items()):
        M = cargar(ram).merge(cargar(arc), on=k, suffixes=("_ram", "_arc"))
        M["cociente"] = M.mediana_s_arc / M.mediana_s_ram
        pmax = M.procesos.max()
        for alg, etiqueta, col in ALGS:
            for p, ls in [(1, ":"), (pmax, "-")]:
                d = M[(M.algoritmo == alg) & (M.procesos == p)].sort_values("n")
                a.plot(d.n, d.cociente, ls, marker="o", ms=3, color=col, label=f"{etiqueta}, p={p}")
        a.axhline(1, ls="--", color="#5f5e58"); a.set_xscale("log")
        a.set_xticks(NS); a.set_xticklabels(NS); a.minorticks_off()
        a.set_xlabel("n"); a.set_title(eq); a.legend(fontsize=6.5, ncol=2)
    ax[0].set_ylabel("T(archivo) / T(RAM), producto")
    fig.tight_layout(); fig.savefig(OUT + "entrada_producto_archivo_vs_ram.png", bbox_inches="tight"); plt.close()


def main():
    datos = EQUIPOS
    t = tabla(datos)
    t.to_csv(R + "comparacion_entrada_archivo.csv", index=False, float_format="%.6g")
    fig_entrada(datos, t); fig_amdahl(datos); fig_producto(datos)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(t.round(3).to_string())


if __name__ == "__main__":
    main()

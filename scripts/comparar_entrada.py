#!/usr/bin/env python3

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
    from matplotlib.lines import Line2D
    ram, arc, c, mk = datos["EPYC (48 cores)"]
    mr, ma = cargar(ram), cargar(arc)
    nombre = dict((a, l) for a, l, _ in ALGS)
    estilo = {BLOQ: ("#D55E00", "s"), PDG: ("#0072B2", "^")}   # color y marcador por algoritmo (Okabe-Ito)
    origen = [(mr, "--", "RAM"), (ma, "-", "archivo")]          # trazos: RAM, continuo: archivo
    gris = "#5f5e58"
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.4))
    n0 = 10000
    for alg, (col, mkr) in estilo.items():
        for m, ls, lab in origen:
            d = m[(m.algoritmo == alg) & (m.n == n0)].sort_values("procesos")
            ax[0].plot(d.procesos, d.speedup_con_generacion_mejor_serial, ls, marker=mkr, ms=4, color=col)
    for m, ls, lab in origen:
        cota = entrada_por_n(m).loc[n0, "cota_amdahl"]
        ax[0].axhline(cota, ls=ls, color=gris, lw=1)
        ax[0].text(48, cota * 1.04, f"cota de Amdahl, {lab}: {cota:.1f}".replace(".", ","), fontsize=7, color=gris, va="bottom", ha="right")
    ax[0].set_ylim(0, 55); ax[0].set_xlabel("procesos p"); ax[0].set_ylabel("speedup de la corrida completa")
    ax[0].set_title(f"n = {n0:,}".replace(",", " ") + ": efecto de p")
    for alg, (col, mkr) in estilo.items():
        for m, ls, lab in origen:
            d = m[(m.algoritmo == alg) & (m.procesos == 48)].sort_values("n")
            ax[1].plot(d.n, d.speedup_con_generacion_mejor_serial, ls, marker=mkr, ms=4, color=col)
    for m, ls, lab in origen:
        e = entrada_por_n(m)
        ax[1].plot(e.index, e.cota_amdahl, ls, color=gris, lw=1)
    ax[1].set_yscale("log"); ax[1].set_xscale("log")
    ax[1].set_xticks(NS); ax[1].set_xticklabels(NS); ax[1].minorticks_off()
    ax[1].set_yticks([1, 2, 5, 10, 20, 50]); ax[1].set_yticklabels([1, 2, 5, 10, 20, 50])
    ax[1].set_xlabel("n"); ax[1].set_ylabel("speedup de la corrida completa"); ax[1].set_title("p = 48: efecto de n")
    h = [Line2D([], [], color=col, marker=mkr, ls="-", ms=5, label=nombre[alg]) for alg, (col, mkr) in estilo.items()]
    h += [Line2D([], [], color=gris, ls="--", label="entrada generada en RAM"),
          Line2D([], [], color=gris, ls="-", label="entrada leída de archivo"),
          Line2D([], [], color=gris, lw=1, label="línea gris fina: cota de Amdahl 1/f")]
    fig.legend(handles=h, loc="lower center", ncol=5, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(OUT + "entrada_amdahl.png", bbox_inches="tight"); plt.close()


def main():
    datos = EQUIPOS
    t = tabla(datos)
    t.to_csv(R + "comparacion_entrada_archivo.csv", index=False, float_format="%.6g")
    fig_entrada(datos, t); fig_amdahl(datos)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(t.round(3).to_string())


if __name__ == "__main__":
    main()

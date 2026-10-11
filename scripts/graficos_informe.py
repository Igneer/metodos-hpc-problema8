import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
R="resultados/"
OUT="img/"
plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,
 "grid.color":"#e3e2dc","figure.dpi":150,"axes.edgecolor":"#5f5e58","legend.frameon":False})
COL={"ijk":"#8a8a85","ikj":"#2a78d6","ikj_bloqueado":"#eb6834"}
LAB={"ijk":"ijk (sin optimizar)","ikj":"ikj (bucles reordenados)","ikj_bloqueado":"ikj con blocking"}
s=pd.read_csv(R+"comparacion_matmul_serial.csv")
fig,ax=plt.subplots(1,2,figsize=(10,4))
for a in ["ijk","ikj","ikj_bloqueado"]:
    d=s[s.algoritmo==a]
    ax[0].errorbar(d.n,d.mediana_segundos,yerr=d.desvio_segundos,marker="o",color=COL[a],label=LAB[a],capsize=3)
    ax[1].plot(d.n,d.mediana_gflops,marker="o",color=COL[a],label=LAB[a])
ax[0].set_xscale("log");ax[0].set_yscale("log");ax[0].set_xlabel("n (matrices n×n)");ax[0].set_ylabel("tiempo (s), mediana de 10 corridas")
ax[0].set_xticks([1000,2000,3000,5000]);ax[0].set_xticklabels([1000,2000,3000,5000]);ax[0].minorticks_off()
ax[0].set_title("Tiempo de ejecución")
ax[1].set_xlabel("n");ax[1].set_ylabel("GFLOPS");ax[1].set_title("Rendimiento");ax[1].legend()
fig.suptitle("Versiones seriales (16 cores, 24 GB de RAM)",y=1.02);fig.tight_layout();fig.savefig(OUT+"serial_tres_versiones.png",bbox_inches="tight");plt.close()

m=pd.read_csv(R+"servidor/metricas.csv")
EST={"mpi_ikj":("#2a78d6","o","1D · ikj"),"mpi_ikj_bloqueado":("#eb6834","s","1D · ikj con blocking"),"ref_pdgemm":("#1baf7a","^","2D · ScaLAPACK PDGEMM")}
def por_p(col,fname,ylabel,title,ideal=False,ns=(1000,5000,10000)):
    fig,ax=plt.subplots(1,len(ns),figsize=(11,3.8),sharey=(not ideal))
    for a,n in zip(ax,ns):
        for alg,(c,mk,l) in EST.items():
            d=m[(m.algoritmo==alg)&(m.n==n)].sort_values("procesos")
            a.plot(d.procesos,d[col],marker=mk,color=c,label=l)
        if ideal: a.plot([1,48],[1,48],"--",color="#5f5e58",label="ideal")
        else: a.axhline(1,ls="--",color="#5f5e58")
        a.set_title(f"n = {n}");a.set_xlabel("procesos p")
    ax[0].set_ylabel(ylabel);ax[0].legend(fontsize=8)
    fig.suptitle(title,y=1.02);fig.tight_layout();fig.savefig(OUT+fname,bbox_inches="tight");plt.close()
por_p("speedup_p1","speedup_vs_p.png","speedup relativo a p = 1","Speedup relativo al propio algoritmo con un proceso (servidor, 48 cores)",ideal=True)
por_p("eficiencia_p1","eficiencia_vs_p.png","eficiencia relativa = S_rel/p","Eficiencia relativa al propio algoritmo con un proceso (servidor, 48 cores)")
por_p("speedup_mejor_serial","speedup_abs_vs_p.png","speedup absoluto = T_serial/T(p)","Speedup absoluto respecto del mejor serial (ikj con blocking) (servidor, 48 cores)",ideal=True)

for col,fname,yl,tt in [("speedup_mejor_serial","speedup_abs_vs_n.png","speedup absoluto = T_serial/T(p)","Efecto del tamaño del problema sobre el speedup absoluto")]:
    fig,ax=plt.subplots(1,2,figsize=(10,3.8),sharey=True)
    for a,p in zip(ax,[16,48]):
        for alg,(c,mk,l) in EST.items():
            d=m[(m.algoritmo==alg)&(m.procesos==p)].sort_values("n")
            a.plot(d.n,d[col],marker=mk,color=c,label=l)
        if col.startswith("speedup"): a.axhline(p,ls="--",color="#5f5e58",label=f"ideal (S = {p})")
        a.set_xscale("log");a.set_xlabel("n");a.set_title(f"p = {p}")
        a.set_xticks([500,1000,2000,5000,10000]);a.set_xticklabels([500,1000,2000,5000,10000])
    ax[0].set_ylabel(yl);ax[0].legend(fontsize=8)
    fig.suptitle(tt,y=1.02);fig.tight_layout();fig.savefig(OUT+fname,bbox_inches="tight");plt.close()

fig,ax=plt.subplots(1,2,figsize=(10,3.8))
for alg,(c,mk,l) in EST.items():
    for n,ls in [(1000,":"),(10000,"-")]:
        d=m[(m.algoritmo==alg)&(m.n==n)&(m.procesos>1)].sort_values("procesos")
        ax[0].plot(d.procesos,d.fraccion_serial_karp_flatt_total,marker=mk,color=c,ls=ls,label=f"{l}, n={n}")
ax[0].set_xlabel("procesos p");ax[0].set_ylabel("fracción serial experimental e");ax[0].set_title("Karp-Flatt (con la generación de A y B)");ax[0].legend(fontsize=6.5)
d=m[(m.algoritmo=="mpi_ikj_bloqueado")&(m.procesos==16)].sort_values("n")
ax[1].plot(d.n,d.speedup_mejor_serial,marker="s",color="#eb6834",label="solo producto + comunicación")
ax[1].plot(d.n,d.speedup_con_generacion_mejor_serial,marker="s",color="#a33",ls="--",label="incluyendo generación serial de A y B")
ax[1].axhline(16,ls="--",color="#5f5e58",label="ideal (p = 16)")
ax[1].set_xscale("log");ax[1].set_xticks([500,1000,2000,5000,10000]);ax[1].set_xticklabels([500,1000,2000,5000,10000])
ax[1].set_xlabel("n");ax[1].set_ylabel("speedup frente al mejor serial");ax[1].set_title("1D · ikj con blocking, p = 16");ax[1].set_ylim(0,18);ax[1].legend(fontsize=8,loc="lower right")
fig.tight_layout();fig.savefig(OUT+"amdahl_karp_flatt.png",bbox_inches="tight");plt.close()

fig,ax=plt.subplots(figsize=(6,3.8))
for alg,(c,mk,l) in EST.items():
    d=m[(m.algoritmo==alg)&(m.n==10000)].sort_values("procesos")
    ax.plot(d.procesos,d.gflops,marker=mk,color=c,label=l)
ax.set_xlabel("procesos p");ax.set_ylabel("GFLOPS");ax.set_title("Rendimiento absoluto, n = 10000");ax.legend(fontsize=8)
fig.tight_layout();fig.savefig(OUT+"gflops_n10000.png",bbox_inches="tight");plt.close()

#!/usr/bin/env python3

import csv
import statistics
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
SRC_COMUN = RAIZ / "src" / "common"
SRC_SERIAL = RAIZ / "src" / "serial"
BIN_DIR = RAIZ / "bin"
BIN_DIR.mkdir(exist_ok=True)

ALGORITMOS = {
    "ikj": SRC_SERIAL / "matmul_ikj.c",
    "ijk": SRC_SERIAL / "matmul_ijk.c",
    "ikj_bloqueado": SRC_SERIAL / "matmul_ikj_bloqueado.c",
}

TAMANIOS = [1000, 2000, 3000, 5000]
REPETICIONES = 10
SEMILLA = 42

CFLAGS = ["-std=gnu11", "-O3", "-march=native"]


def compilar(algoritmo, fuente, n):
    binario = BIN_DIR / f"matmul_{algoritmo}_{n}.exe"
    cmd = [
        "gcc", *CFLAGS,
        f"-DN={n}", f"-DM={n}", f"-DP={n}",
        str(fuente), str(SRC_COMUN / "matrix.c"), str(SRC_COMUN / "cli.c"),
        "-o", str(binario),
    ]
    resultado = subprocess.run(cmd, capture_output=True, text=True)
    if resultado.returncode != 0:
        print(f"ERROR compilando {algoritmo} n={n}:\n{resultado.stderr}", file=sys.stderr)
        sys.exit(1)
    return binario


def correr(binario):
    cmd = [str(binario), "--semilla", str(SEMILLA),
           "--repeticiones", str(REPETICIONES), "--csv"]
    resultado = subprocess.run(cmd, capture_output=True, text=True)
    if resultado.returncode != 0:
        print(f"ERROR corriendo {binario}:\n{resultado.stderr}", file=sys.stderr)
        sys.exit(1)
    lineas = resultado.stdout.strip().splitlines()
    lector = csv.DictReader(lineas)
    segundos = []
    gflops = []
    for fila in lector:
        segundos.append(float(fila["segundos"]))
        gflops.append(float(fila["gflops"]))
    return segundos, gflops


def main():
    filas_resumen = []
    total = len(ALGORITMOS) * len(TAMANIOS)
    hecho = 0
    for algoritmo, fuente in ALGORITMOS.items():
        for n in TAMANIOS:
            hecho += 1
            print(f"[{hecho}/{total}] Compilando y corriendo {algoritmo} "
                  f"N=M=P={n} ({REPETICIONES} repeticiones)...", flush=True)
            binario = compilar(algoritmo, fuente, n)
            segundos, gflops = correr(binario)
            mediana_s = statistics.median(segundos)
            desvio_s = statistics.stdev(segundos) if len(segundos) > 1 else 0.0
            mediana_gflops = statistics.median(gflops)
            filas_resumen.append({
                "algoritmo": algoritmo,
                "n": n,
                "m": n,
                "p": n,
                "repeticiones": REPETICIONES,
                "mediana_segundos": f"{mediana_s:.9f}",
                "desvio_segundos": f"{desvio_s:.9f}",
                "mediana_gflops": f"{mediana_gflops:.4f}",
            })
            print(f"    mediana = {mediana_s:.4f} s ({mediana_gflops:.2f} GFLOPS)",
                  flush=True)

    salida = RAIZ / "comparacion_matmul_serial.csv"
    campos = ["algoritmo", "n", "m", "p", "repeticiones",
              "mediana_segundos", "desvio_segundos", "mediana_gflops"]
    with open(salida, "w", newline="") as f:
        escritor = csv.DictWriter(f, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(filas_resumen)

    print(f"\nListo. Resultados en {salida}")


if __name__ == "__main__":
    main()

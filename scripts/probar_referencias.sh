#!/usr/bin/env bash
# Compila y prueba las referencias de biblioteca (OpenBLAS dgemm y
# ScaLAPACK PDGEMM 2D) con N=M=P=${1:-500}. Correr en WSL.
# PDGEMM verifica que C == 2*N; si falla, aborta.
set -euo pipefail
cd "$(dirname "$0")/.."
N=${1:-500}
mkdir -p bin
F="-std=gnu11 -Wall -Wextra -O3 -march=native -DN=$N -DM=$N -DP=$N"
C="src/common/matrix.c src/common/cli.c"s
mpicc $F src/ref/matmul_pdgemm.c $C -o bin/ref_pdgemm -lscalapack-openmpi -lopenblas -lm
./bin/ref_dgemm --repeticiones 3
for p in 1 2 3 4; do
  mpirun --oversubscribe -np $p ./bin/ref_pdgemm --repeticiones 3
done
echo "OK"

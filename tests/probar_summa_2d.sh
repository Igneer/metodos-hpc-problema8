#!/bin/sh
# Compila y ejecuta la prueba de SUMMA 2D con varias dimensiones y procesos
# (grillas no cuadradas, cortes desiguales y bloques vacios). Requiere mpicc/mpirun.
set -e
cd "$(dirname "$0")/.."
salida="${TMPDIR:-/tmp}/test_summa_2d"
for dims in "5 6 7" "4 4 4" "3 2 9" "7 5 3" "1 1 1" "16 16 16"; do
    set -- $dims
    mpicc -std=c11 -Wall -Wextra -Wno-comment -DN=$1 -DM=$2 -DP=$3 tests/test_summa_2d.c \
        src/mpi/2d/*.c src/common/particiones.c -o "$salida"
    for np in 1 2 3 4 5 6 8 9 12; do
        mpirun --oversubscribe -np $np "$salida"
    done
done

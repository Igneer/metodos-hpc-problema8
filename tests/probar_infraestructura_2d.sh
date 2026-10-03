#!/bin/sh
# Compila y ejecuta la prueba de la etapa 2 con varias dimensiones y procesos
# (incluye grillas no cuadradas y bloques vacios). Requiere mpicc/mpirun.
set -e
cd "$(dirname "$0")/.."
salida="${TMPDIR:-/tmp}/test_infra_2d"
for dims in "5 6 7" "4 4 4" "3 2 9" "1 1 1"; do
    set -- $dims
    mpicc -std=c11 -Wall -Wextra -DN=$1 -DM=$2 -DP=$3 tests/test_infraestructura_2d.c \
        src/mpi/2d/*.c src/common/particiones.c -o "$salida"
    for np in 1 2 3 4 6 8; do
        mpirun --oversubscribe -np $np "$salida"
    done
done

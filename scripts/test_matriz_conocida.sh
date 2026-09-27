#!/usr/bin/env bash
#
# test_matriz_conocida.sh
#
# Compila las 8 implementaciones de multiplicacion de matrices (3
# seriales, 3 MPI row-wise, 2 SUMMA) con N=M=P=3, las corre con la
# matriz de tests/datos/matriz_conocida.txt como A y B, e imprime C
# comparandolo contra A*A calculado a mano. Sirve para verificar que
# el reparto/recoleccion de cada estrategia no tenga bugs de indices
# antes de confiar en un benchmark de rendimiento mas grande.
#
# Uso:
#   bash scripts/test_matriz_conocida.sh
#
# Requiere gcc, mpicc y mpirun en el PATH.

set -uo pipefail

RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$RAIZ"

for herramienta in gcc mpicc mpirun; do
    if ! command -v "$herramienta" >/dev/null 2>&1; then
        echo "Error: no se encontro '$herramienta' en el PATH." >&2
        exit 1
    fi
done

BIN_DIR="$RAIZ/bin"
mkdir -p "$BIN_DIR"

MATRIZ="$RAIZ/tests/datos/matriz_conocida.txt"
CFLAGS="-std=c11 -O2 -DN=3 -DM=3 -DP=3"
ARGS=(--archivo-a "$MATRIZ" --archivo-b "$MATRIZ" --repeticiones 1 --imprimir)

# C = A*A calculada a mano para la matriz de matriz_conocida.txt
# (2 1 4 / 3 2 1 / 1 2 2), fila por fila.
ESPERADO=("11 12 17" "13 9 16" "10 9 10")

TOTAL=0
FALLOS=0


# Compila en el shell actual (no en una subshell de command substitution:
# ahi un `exit`/`return` no frena el script que llama) y corta la
# ejecucion mostrando el log si falla.
compilar() {
    local compilador="$1" nombre="$2"; shift 2
    local destino="$BIN_DIR/$nombre"
    if ! "$compilador" $CFLAGS "$@" -o "$destino" 2>"$BIN_DIR/$nombre.compile.log"; then
        echo "ERROR compilando $nombre:" >&2
        cat "$BIN_DIR/$nombre.compile.log" >&2
        exit 1
    fi
}

# Compara las ultimas 3 lineas de $salida (la matriz C impresa por
# matriz_imprimir) contra $ESPERADO, valor por valor con tolerancia.
comparar_salida() {
    local nombre="$1" salida="$2"
    TOTAL=$((TOTAL + 1))

    local ultimas
    ultimas="$(printf '%s\n' "$salida" | tail -n 3)"

    local ok=1 filas=0
    while IFS= read -r linea_real; do
        local esperada="${ESPERADO[$filas]:-}"
        if [ -z "$esperada" ]; then
            ok=0
            break
        fi
        read -r -a vals_reales <<< "$linea_real"
        read -r -a vals_esperados <<< "$esperada"
        for j in 0 1 2; do
            local a="${vals_reales[$j]:-}" b="${vals_esperados[$j]}"
            local diff
            diff=$(awk -v a="$a" -v b="$b" 'BEGIN {
                if (a == "") { print "bad"; exit }
                d = a - b; if (d < 0) d = -d;
                print (d < 1e-6) ? "ok" : "bad"
            }')
            [ "$diff" = "ok" ] || ok=0
        done
        filas=$((filas + 1))
    done <<< "$ultimas"

    if [ "$ok" -eq 1 ] && [ "$filas" -eq 3 ]; then
        echo "[OK]   $nombre"
    else
        FALLOS=$((FALLOS + 1))
        echo "[FAIL] $nombre"
        printf '%s\n' "$salida" | sed 's/^/       /'
    fi
}

echo "=== Compilando (N=M=P=3) ==="
compilar gcc matmul_ijk_s src/serial/matmul_ijk.c src/common/matrix.c src/common/cli.c
compilar gcc matmul_ikj_s src/serial/matmul_ikj.c src/common/matrix.c src/common/cli.c
compilar gcc matmul_ikjb_s src/serial/matmul_ikj_bloqueado.c src/common/matrix.c src/common/cli.c

compilar mpicc matmul_ijk_mpi src/mpi/matmul_ijk.c src/common/matrix.c src/common/cli.c src/common/particiones.c
compilar mpicc matmul_ikj_mpi src/mpi/matmul_ikj.c src/common/matrix.c src/common/cli.c src/common/particiones.c
compilar mpicc matmul_ikjb_mpi src/mpi/matmul_ikj_bloqueado.c src/common/matrix.c src/common/cli.c src/common/particiones.c

compilar mpicc matmul_summa src/mpi/matmul_summa.c src/common/matrix.c src/common/cli.c src/common/particiones.c

BIN_IJK_S="$BIN_DIR/matmul_ijk_s"
BIN_IKJ_S="$BIN_DIR/matmul_ikj_s"
BIN_IKJB_S="$BIN_DIR/matmul_ikjb_s"
BIN_IJK_MPI="$BIN_DIR/matmul_ijk_mpi"
BIN_IKJ_MPI="$BIN_DIR/matmul_ikj_mpi"
BIN_IKJB_MPI="$BIN_DIR/matmul_ikjb_mpi"
BIN_SUMMA="$BIN_DIR/matmul_summa"

echo ""
echo "=== Corriendo ==="

comparar_salida "serial ijk" "$("$BIN_IJK_S" "${ARGS[@]}")"
comparar_salida "serial ikj" "$("$BIN_IKJ_S" "${ARGS[@]}")"
comparar_salida "serial ikj_bloqueado" "$("$BIN_IKJB_S" "${ARGS[@]}")"

for np in 1 3; do
    comparar_salida "mpi ijk (np=$np)" "$(mpirun -np "$np" "$BIN_IJK_MPI" "${ARGS[@]}")"
    comparar_salida "mpi ikj (np=$np)" "$(mpirun -np "$np" "$BIN_IKJ_MPI" "${ARGS[@]}")"
    comparar_salida "mpi ikj_bloqueado (np=$np)" "$(mpirun -np "$np" "$BIN_IKJB_MPI" "${ARGS[@]}")"
done

# SUMMA acepta cualquier np (grilla Pr x Pc factorizada, no hace
# falta cuadrado perfecto): probamos np=2 y np=3 (grillas no
# cuadradas) ademas de los cuadrados 1 y 4.
for np in 1 2 3 4; do
    comparar_salida "summa (np=$np)" "$(mpirun -np "$np" "$BIN_SUMMA" "${ARGS[@]}")"
done

echo ""
echo "=== Resumen: $((TOTAL - FALLOS))/$TOTAL OK ==="
[ "$FALLOS" -eq 0 ]

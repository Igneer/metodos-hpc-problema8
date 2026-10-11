#!/usr/bin/env bash

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

MATRIZ_A="$RAIZ/tests/datos/matriz_a_4x4.txt"
MATRIZ_B="$RAIZ/tests/datos/matriz_b_4x4.txt"
CFLAGS="-std=gnu11 -O3 -march=native -DN=4 -DM=4 -DP=4"
ARGS=(--archivo-a "$MATRIZ_A" --archivo-b "$MATRIZ_B" --repeticiones 1 --imprimir)

# C = A*B calculada a mano, fila por fila. Ejemplo, C[0][0] =
# 1*16 + 2*12 + 3*8 + 4*4 = 80.
ESPERADO=("80 70 60 50" "240 214 188 162" "400 358 316 274" "560 502 444 386")

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

# Compara las ultimas 4 lineas de $salida (la matriz C impresa por
# matriz_imprimir) contra $ESPERADO, valor por valor con tolerancia.
comparar_salida() {
    local nombre="$1" salida="$2"
    TOTAL=$((TOTAL + 1))

    local ultimas
    ultimas="$(printf '%s\n' "$salida" | tail -n 4)"

    local ok=1 filas=0
    while IFS= read -r linea_real; do
        local esperada="${ESPERADO[$filas]:-}"
        if [ -z "$esperada" ]; then
            ok=0
            break
        fi
        read -r -a vals_reales <<< "$linea_real"
        read -r -a vals_esperados <<< "$esperada"
        for j in 0 1 2 3; do
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

    if [ "$ok" -eq 1 ] && [ "$filas" -eq 4 ]; then
        echo "[OK]   $nombre"
    else
        FALLOS=$((FALLOS + 1))
        echo "[FAIL] $nombre"
        printf '%s\n' "$salida" | sed 's/^/       /'
    fi
}

echo "=== Compilando (N=M=P=4) ==="
compilar gcc matmul_ijk_s src/serial/matmul_ijk.c src/common/matrix.c src/common/cli.c
compilar gcc matmul_ikj_s src/serial/matmul_ikj.c src/common/matrix.c src/common/cli.c
compilar gcc matmul_ikjb_s src/serial/matmul_ikj_bloqueado.c src/common/matrix.c src/common/cli.c

compilar mpicc matmul_ijk_mpi src/mpi/matmul_ijk.c src/common/matrix.c src/common/cli.c src/common/particiones.c
compilar mpicc matmul_ikj_mpi src/mpi/matmul_ikj.c src/common/matrix.c src/common/cli.c src/common/particiones.c
compilar mpicc matmul_ikjb_mpi src/mpi/matmul_ikj_bloqueado.c src/common/matrix.c src/common/cli.c src/common/particiones.c

# Con N=4 el bloque por defecto (64) cubre toda la matriz y el kernel bloqueado
# hace una sola vuelta. Se prueban bloques que si la parten: 2 (divide a 4),
# 3 (no divide: el ultimo bloque es de 1 elemento) y tamanios distintos por
# dimension.
BLOQUES_ETIQ=("BLOQUE=2" "BLOQUE=3" "BI=2 BK=3 BJ=1")
BLOQUES_FLAGS=("-DBLOQUE=2" "-DBLOQUE=3" "-DBI=2 -DBK=3 -DBJ=1")
for i in "${!BLOQUES_FLAGS[@]}"; do
    compilar gcc "matmul_ikjb_s_$i" ${BLOQUES_FLAGS[$i]} src/serial/matmul_ikj_bloqueado.c src/common/matrix.c src/common/cli.c
    compilar mpicc "matmul_ikjb_mpi_$i" ${BLOQUES_FLAGS[$i]} src/mpi/matmul_ikj_bloqueado.c src/common/matrix.c src/common/cli.c src/common/particiones.c
done

# La referencia ScaLAPACK es opcional: si no se puede enlazar, se omite.
# Se compila con varios tamanios de bloque NB (con N=4: bloques de 1, 2 y 3
# elementos que reparten la matriz entre varios procesos, y el valor por
# defecto 64, mayor que N: un solo bloque y los demas procesos sin datos).
NBS=(1 2 3 64)
PDGEMM_OK=1
for nb in "${NBS[@]}"; do
    if ! mpicc $CFLAGS -DNB="$nb" src/ref/matmul_pdgemm.c src/common/matrix.c src/common/cli.c             -o "$BIN_DIR/matmul_pdgemm_nb$nb" -lscalapack-openmpi -lopenblas -lm             2>"$BIN_DIR/matmul_pdgemm_nb$nb.compile.log"; then
        PDGEMM_OK=0
        echo "AVISO: no se pudo compilar la referencia PDGEMM con NB=$nb (ver bin/matmul_pdgemm_nb$nb.compile.log); se omite."
        break
    fi
done

BIN_IJK_S="$BIN_DIR/matmul_ijk_s"
BIN_IKJ_S="$BIN_DIR/matmul_ikj_s"
BIN_IKJB_S="$BIN_DIR/matmul_ikjb_s"
BIN_IJK_MPI="$BIN_DIR/matmul_ijk_mpi"
BIN_IKJ_MPI="$BIN_DIR/matmul_ikj_mpi"
BIN_IKJB_MPI="$BIN_DIR/matmul_ikjb_mpi"

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

# ikj bloqueado con bloques que si parten la matriz (serial y MPI).
for i in "${!BLOQUES_FLAGS[@]}"; do
    etiqueta="${BLOQUES_ETIQ[$i]}"
    comparar_salida "serial ikj_bloqueado ($etiqueta)" "$("$BIN_DIR/matmul_ikjb_s_$i" "${ARGS[@]}")"
    for np in 1 3; do
        comparar_salida "mpi ikj_bloqueado ($etiqueta, np=$np)" "$(mpirun -np "$np" "$BIN_DIR/matmul_ikjb_mpi_$i" "${ARGS[@]}")"
    done
done

# Grillas 2x2 (np=4), 3x2 (np=6) y 4x2 (np=8); con N=4 algunas tienen
# bloques vacios. Cada una se prueba con todos los NB compilados.
export OPENBLAS_NUM_THREADS=1
if [ "$PDGEMM_OK" -eq 1 ]; then
    for nb in "${NBS[@]}"; do
        for np in 1 2 3 4 6 8; do
            comparar_salida "scalapack pdgemm (NB=$nb, np=$np)" "$(mpirun --oversubscribe -np "$np" "$BIN_DIR/matmul_pdgemm_nb$nb" "${ARGS[@]}")"
        done
    done
fi

echo ""
echo "=== Resumen: $((TOTAL - FALLOS))/$TOTAL OK ==="
[ "$FALLOS" -eq 0 ]

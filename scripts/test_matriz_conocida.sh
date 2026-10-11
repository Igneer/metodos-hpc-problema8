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
DATOS="$RAIZ/tests/datos"
mkdir -p "$BIN_DIR"

# Caso: etiqueta N M P archivo_A archivo_B archivo_C_esperada
CASOS=(
    "4x4x4 4 4 4 matriz_a_4x4.txt matriz_b_4x4.txt matriz_c_4x4.txt"
    "5x3x7 5 3 7 matriz_a_5x3.txt matriz_b_3x7.txt matriz_c_5x7.txt"
    "2x6x3 2 6 3 matriz_a_2x6.txt matriz_b_6x3.txt matriz_c_2x3.txt"
)

# Con matrices tan chicas el bloque por defecto (64) cubre toda la matriz y el
# kernel bloqueado hace una sola vuelta. Se prueban bloques que si la parten:
# 2, 3 (que no divide a 4, 5 ni 7: queda un ultimo bloque incompleto) y
# tamanios distintos por dimension.
BLOQUES_ETIQ=("BLOQUE=2" "BLOQUE=3" "BI=2 BK=3 BJ=1")
BLOQUES_FLAGS=("-DBLOQUE=2" "-DBLOQUE=3" "-DBI=2 -DBK=3 -DBJ=1")

# Tamanios de bloque NB de la referencia ScaLAPACK: 1, 2 y 3 reparten la matriz
# entre varios procesos; 64 es mayor que la matriz (un solo bloque y los demas
# procesos sin datos).
NBS=(1 2 3 64)

TOTAL=0
FALLOS=0
export OPENBLAS_NUM_THREADS=1

# Compila en el shell actual (no en una subshell de command substitution:
# ahi un `exit`/`return` no frena el script que llama) y corta la
# ejecucion mostrando el log si falla. Usa $CFLAGS del caso en curso.
compilar() {
    local compilador="$1" nombre="$2"; shift 2
    local destino="$BIN_DIR/$nombre"
    if ! "$compilador" $CFLAGS "$@" -o "$destino" 2>"$BIN_DIR/$nombre.compile.log"; then
        echo "ERROR compilando $nombre:" >&2
        cat "$BIN_DIR/$nombre.compile.log" >&2
        exit 1
    fi
}

# Compara las ultimas $N lineas de $salida (la matriz C impresa por
# matriz_imprimir) contra el archivo $ESPERADO, valor por valor con tolerancia.
comparar_salida() {
    local nombre="$1" salida="$2"
    TOTAL=$((TOTAL + 1))

    if printf '%s\n' "$salida" | tail -n "$N" | awk -v ne="$N" -v pe="$P" '
        NR == FNR { for (j = 1; j <= NF; j++) e[FNR, j] = $j; next }
        {
            if (NF != pe) bad = 1
            for (j = 1; j <= NF; j++) {
                d = $j - e[FNR, j]; if (d < 0) d = -d
                if (d > 1e-6) bad = 1
            }
            filas++
        }
        END { exit !(filas == ne && !bad) }' "$ESPERADO" -; then
        echo "[OK]   $nombre"
    else
        FALLOS=$((FALLOS + 1))
        echo "[FAIL] $nombre"
        printf '%s\n' "$salida" | sed 's/^/       /'
    fi
}

probar_caso() {
    local etiqueta="$1"
    N=$2; M=$3; P=$4
    local A="$DATOS/$5" B="$DATOS/$6"
    ESPERADO="$DATOS/$7"
    CFLAGS="-std=gnu11 -O3 -march=native -DN=$N -DM=$M -DP=$P"
    local ARGS=(--archivo-a "$A" --archivo-b "$B" --repeticiones 1 --imprimir)
    local COM="src/common/matrix.c src/common/cli.c"
    local PART="src/common/particiones.c"
    local sufijo="_$etiqueta"

    echo ""
    echo "=== Caso $etiqueta (N=$N M=$M P=$P): compilando ==="
    compilar gcc "matmul_ijk_s$sufijo" src/serial/matmul_ijk.c $COM
    compilar gcc "matmul_ikj_s$sufijo" src/serial/matmul_ikj.c $COM
    compilar gcc "matmul_ikjb_s$sufijo" src/serial/matmul_ikj_bloqueado.c $COM
    compilar mpicc "matmul_ijk_mpi$sufijo" src/mpi/matmul_ijk.c $COM $PART
    compilar mpicc "matmul_ikj_mpi$sufijo" src/mpi/matmul_ikj.c $COM $PART
    compilar mpicc "matmul_ikjb_mpi$sufijo" src/mpi/matmul_ikj_bloqueado.c $COM $PART

    local i nb np
    for i in "${!BLOQUES_FLAGS[@]}"; do
        compilar gcc "matmul_ikjb_s_$i$sufijo" ${BLOQUES_FLAGS[$i]} src/serial/matmul_ikj_bloqueado.c $COM
        compilar mpicc "matmul_ikjb_mpi_$i$sufijo" ${BLOQUES_FLAGS[$i]} src/mpi/matmul_ikj_bloqueado.c $COM $PART
    done

    # La referencia ScaLAPACK es opcional: si no se puede enlazar, se omite.
    local pdgemm_ok=1
    for nb in "${NBS[@]}"; do
        if ! mpicc $CFLAGS -DNB="$nb" src/ref/matmul_pdgemm.c $COM \
                -o "$BIN_DIR/matmul_pdgemm_nb$nb$sufijo" -lscalapack-openmpi -lopenblas -lm \
                2>"$BIN_DIR/matmul_pdgemm_nb$nb$sufijo.compile.log"; then
            pdgemm_ok=0
            echo "AVISO: no se pudo compilar la referencia PDGEMM con NB=$nb (ver bin/matmul_pdgemm_nb$nb$sufijo.compile.log); se omite."
            break
        fi
    done

    echo "=== Caso $etiqueta: corriendo ==="
    comparar_salida "[$etiqueta] serial ijk" "$("$BIN_DIR/matmul_ijk_s$sufijo" "${ARGS[@]}")"
    comparar_salida "[$etiqueta] serial ikj" "$("$BIN_DIR/matmul_ikj_s$sufijo" "${ARGS[@]}")"
    comparar_salida "[$etiqueta] serial ikj_bloqueado" "$("$BIN_DIR/matmul_ikjb_s$sufijo" "${ARGS[@]}")"

    for np in 1 3; do
        comparar_salida "[$etiqueta] mpi ijk (np=$np)" "$(mpirun --oversubscribe -np "$np" "$BIN_DIR/matmul_ijk_mpi$sufijo" "${ARGS[@]}")"
        comparar_salida "[$etiqueta] mpi ikj (np=$np)" "$(mpirun --oversubscribe -np "$np" "$BIN_DIR/matmul_ikj_mpi$sufijo" "${ARGS[@]}")"
        comparar_salida "[$etiqueta] mpi ikj_bloqueado (np=$np)" "$(mpirun --oversubscribe -np "$np" "$BIN_DIR/matmul_ikjb_mpi$sufijo" "${ARGS[@]}")"
    done

    # ikj bloqueado con bloques que si parten la matriz (serial y MPI).
    for i in "${!BLOQUES_FLAGS[@]}"; do
        comparar_salida "[$etiqueta] serial ikj_bloqueado (${BLOQUES_ETIQ[$i]})" "$("$BIN_DIR/matmul_ikjb_s_$i$sufijo" "${ARGS[@]}")"
        for np in 1 3; do
            comparar_salida "[$etiqueta] mpi ikj_bloqueado (${BLOQUES_ETIQ[$i]}, np=$np)" \
                "$(mpirun --oversubscribe -np "$np" "$BIN_DIR/matmul_ikjb_mpi_$i$sufijo" "${ARGS[@]}")"
        done
    done

    # Grillas de 1 a 8 procesos; con matrices tan chicas algunas tienen
    # bloques vacios. Cada una se prueba con todos los NB compilados.
    if [ "$pdgemm_ok" -eq 1 ]; then
        for nb in "${NBS[@]}"; do
            for np in 1 2 3 4 6 8; do
                comparar_salida "[$etiqueta] scalapack pdgemm (NB=$nb, np=$np)" \
                    "$(mpirun --oversubscribe -np "$np" "$BIN_DIR/matmul_pdgemm_nb$nb$sufijo" "${ARGS[@]}")"
            done
        done
    fi
}

for caso in "${CASOS[@]}"; do
    read -r -a campos <<< "$caso"
    probar_caso "${campos[@]}"
done

echo ""
echo "=== Resumen: $((TOTAL - FALLOS))/$TOTAL OK ==="
[ "$FALLOS" -eq 0 ]

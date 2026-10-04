#!/usr/bin/env bash
# Barrido del tamanio de bloque del serial ikj bloqueado (BI=BK=BJ=B).
# Deja un CSV  n,bloque,gflops  (media de las repeticiones) para comparar
# con resultados/servidor/barrido_bloque.csv.
#
# Uso:
#   scripts/barrido_bloque.sh                       # defecto: N=1000 2000 4000
#   scripts/barrido_bloque.sh --ns "2000 4000" --bloques "64 128 160" --reps 3
#   scripts/barrido_bloque.sh --salida resultados/local/barrido_bloque.csv
# Conviene correrlo con el equipo enchufado y sin otros programas pesados.
set -uo pipefail
cd "$(dirname "$0")/.."

NS="1000 2000 4000"; BS="32 64 96 128 160 192 256"; REPS=5
SALIDA=resultados/local/barrido_bloque.csv
while [ $# -gt 0 ]; do
  case "$1" in
    --ns) NS=$2; shift 2;;  --bloques) BS=$2; shift 2;;
    --reps) REPS=$2; shift 2;;  --salida) SALIDA=$2; shift 2;;
    *) echo "Argumento desconocido: $1" >&2; exit 1;;
  esac
done

FLAGS="-std=gnu11 -O3 -march=native"
[ "$(uname -s)" = Linux ] && [ "$(uname -m)" = x86_64 ] && FLAGS="$FLAGS -mcmodel=medium"
SRC="src/serial/matmul_ikj_bloqueado.c src/common/matrix.c src/common/cli.c"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

mkdir -p "$(dirname "$SALIDA")"
echo "n,bloque,gflops" > "$SALIDA"
echo "Salida: $SALIDA   N: $NS   B: $BS   reps: $REPS"
for n in $NS; do
  for b in $BS; do
    exe=$TMP/bloq_${n}_$b
    # shellcheck disable=SC2086
    if ! gcc $FLAGS -DN=$n -DM=$n -DP=$n -DBI=$b -DBK=$b -DBJ=$b $SRC -lm -o "$exe"; then
      echo "[ERROR compilar] n=$n b=$b" >&2; continue
    fi
    g=$("$exe" --csv --repeticiones "$REPS" | awk -F, 'NR>1{s+=$7; c++} END{if(c) printf "%.2f", s/c}')
    echo "$n,$b,$g" | tee -a "$SALIDA"
  done
done
echo "Listo. Datos en $SALIDA"

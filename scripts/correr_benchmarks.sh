#!/usr/bin/env bash

set -uo pipefail
cd "$(dirname "$0")/.."

# PERFIL solo etiqueta la corrida (columna "perfil", carpeta de salida y binarios).
PERFIL=servidor; CORES=""; RAM_GB=""; NS=""; PS=""; SALIDA=""; DRY=0; DESDE_ARCHIVO=0; DIR_MATRICES=""
REPS=5; REPS_CHICAS=10; N_CHICO=2000   # 10 repeticiones si n <= 2000, 5 si no
TIMEOUT=21600                          # 6 h por combinacion (todas las repeticiones)
while [ $# -gt 0 ]; do
  case "$1" in
    --perfil) PERFIL=$2; shift 2;;   --cores) CORES=$2; shift 2;;
    --ram-gb) RAM_GB=$2; shift 2;;   --ns) NS=$2; shift 2;;
    --ps) PS=$2; shift 2;;           --salida) SALIDA=$2; shift 2;;
    --desde-archivo) DESDE_ARCHIVO=1; shift;;  --dir-matrices) DIR_MATRICES=$2; shift 2;;
    --dry-run) DRY=1; shift;;
    *) echo "Argumento desconocido: $1" >&2; exit 1;;
  esac
done

# ---------- Hardware ----------
CORES=${CORES:-$(nproc)}
RAM_GB=${RAM_GB:-$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo 2>/dev/null)}
RAM_GB=${RAM_GB:-8}

[ -z "$NS" ] && NS="500 1000 1500 2000 3000 4000 5000 7500 10000"
# Procesos MPI: 1 y potencias de 2 hasta CORES, mas CORES (p=1 sirve de base del
# speedup MPI, aparte del serial). Nunca oversubscribe.
if [ -z "$PS" ]; then
  PS=1; p=2; while [ $p -lt "$CORES" ]; do PS="$PS $p"; p=$((p*2)); done
  [ "$CORES" -gt 1 ] && PS="$PS $CORES"
fi
ALGS="serial_ikj serial_ikj_bloqueado mpi_ikj mpi_ikj_bloqueado ref_pdgemm"

es_serial() { case $1 in serial_*) return 0;; *) return 1;; esac; }
reps_para() { [ "$1" -le "$N_CHICO" ] && echo "$REPS_CHICAS" || echo "$REPS"; }   
rate_gflops() { case $1 in *ikj) echo 6;; *) echo 25;; esac; }   # GFLOPS tipicos, solo para la estimacion

OUT=${SALIDA:-resultados/${PERFIL}_${CORES}c}
[ "$DESDE_ARCHIVO" = 1 ] && [ -z "$SALIDA" ] && OUT=${OUT}_archivo
DIR_MATRICES=${DIR_MATRICES:-tests/datos}
BIN=bin/bench_${PERFIL}
mkdir -p "$OUT" "$BIN"
CSV=$OUT/crudo.csv; HECHOS=$OUT/hechos.txt; FALLOS=$OUT/fallos.txt
[ -f "$CSV" ] || echo "perfil,n,procesos,algoritmo,repeticion,segundos,gflops,segundos_generacion" > "$CSV"
touch "$HECHOS" "$FALLOS"

# ---------- Compilacion ----------
FLAGS="-std=gnu11 -O3 -march=native"
[ "$(uname -m)" = x86_64 ] && FLAGS="$FLAGS -mcmodel=medium"   # BSS > 2 GB
COM="src/common/matrix.c src/common/cli.c"
# Libs de ScaLAPACK/OpenBLAS para ref_pdgemm. Debian/Ubuntu: -lscalapack-openmpi.
# Con conda-forge (libscalapack.so en $CONDA_PREFIX/lib) se autodetecta; o definir
# SCALAPACK_LIBS="..." a mano.
if [ -z "${SCALAPACK_LIBS:-}" ]; then
  if [ -n "${CONDA_PREFIX:-}" ] && ls "$CONDA_PREFIX"/lib/libscalapack.so* >/dev/null 2>&1; then
    SCALAPACK_LIBS="-L$CONDA_PREFIX/lib -Wl,-rpath,$CONDA_PREFIX/lib -lscalapack -lopenblas"
  else
    SCALAPACK_LIBS="-lscalapack-openmpi -lopenblas"
  fi
fi
src_de() { case $1 in
  serial_*) echo "gcc src/serial/matmul_${1#serial_}.c $COM -lm";;
  mpi_*) echo "mpicc src/mpi/matmul_${1#mpi_}.c src/common/particiones.c $COM -lm";;
  ref_pdgemm) echo "mpicc src/ref/matmul_pdgemm.c $COM $SCALAPACK_LIBS -lm";;
esac; }
compilar() { # alg n
  local exe=$BIN/${1}_$2
  local cmd; cmd=$(src_de "$1")
  # Reutiliza el binario solo si es mas nuevo que sus fuentes y headers (uno
  # viejo puede imprimir otro formato de csv y quedar descartado en silencio).
  if [ -x "$exe" ]; then
    local viejo=0 f
    for f in $(echo "$cmd" | tr ' ' '\n' | grep '[.]c$') src/common/*.h; do
      [ "$f" -nt "$exe" ] && viejo=1
    done
    [ $viejo = 0 ] && return 0
  fi
  # El compilador va primero; los flags y -D antes de las fuentes no importan
  # salvo las librerias (-l), que deben ir al final: se arma explicitamente.
  local comp=${cmd%% *} resto=${cmd#* } libs="" fuentes=""
  for t in $resto; do case $t in -l*) libs="$libs $t";; *) fuentes="$fuentes $t";; esac; done
  # shellcheck disable=SC2086
  $comp $FLAGS -DN=$2 -DM=$2 -DP=$2 $fuentes -o "$exe" $libs 2>>"$OUT/compilacion.log"
}
disponible() { # alg
  case $1 in
    mpi_*) command -v mpicc >/dev/null || return 1;;
    ref_pdgemm) command -v mpicc >/dev/null || return 1
                # prueba de enlace real (ldconfig no ve librerias de conda)
                echo 'int main(void){return 0;}' |                   mpicc -x c - -o /dev/null $SCALAPACK_LIBS 2>/dev/null || return 1;;
  esac; return 0
}

# ---------- Matrices en archivo ----------
# Se generan una sola vez por N (mismas semillas que los benchmarks) y se
# comparten entre algoritmos y repeticiones. Antes de cada corrida se vacia su
# cache de paginas (enfriar_cache), asi todas leen del disco en las mismas condiciones.
ARGS_ARCHIVO=""
asegurar_matrices() { # n
  [ -s "$DIR_MATRICES/A_$1.txt" ] && [ -s "$DIR_MATRICES/B_$1.txt" ] && return 0
  mkdir -p "$DIR_MATRICES" "$BIN"
  [ -x "$BIN/generar_matrices" ] || gcc -std=gnu11 -O2 src/tools/generar_matrices.c src/common/matrix.c     -o "$BIN/generar_matrices" 2>>"$OUT/compilacion.log" || return 1
  echo "[generando] matrices n=$1 en $DIR_MATRICES"
  "$BIN/generar_matrices" "$1" "$DIR_MATRICES"
}

# Vacia del cache de paginas los archivos de A y B. Con root usa drop_caches
# (todo el cache); sin root usa dd iflag=nocache (posix_fadvise DONTNEED, solo
# esos archivos y solo paginas limpias).
AVISO_FRIO=0
enfriar_cache() { # n
  sync
  if [ -w /proc/sys/vm/drop_caches ] && echo 3 > /proc/sys/vm/drop_caches 2>/dev/null; then return 0; fi
  if sudo -n true 2>/dev/null && echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1; then return 0; fi
  local f ok=1
  for f in "$DIR_MATRICES/A_$1.txt" "$DIR_MATRICES/B_$1.txt"; do
    dd if="$f" iflag=nocache count=0 status=none 2>/dev/null || ok=0
  done
  [ $ok = 1 ] && return 0
  if [ $AVISO_FRIO = 0 ]; then
    echo "AVISO: no se pudo vaciar el cache (sin root ni dd iflag=nocache); las lecturas pueden venir de RAM." >&2
    AVISO_FRIO=1
  fi
  return 0
}

# ---------- Plan ----------
HOST=$(hostname)
echo "Perfil=$PERFIL cores=$CORES RAM=${RAM_GB}GB"
echo "N: $NS"; echo "P: $PS"; echo "Salida: $CSV"
{ echo "fecha: $(date -Is)"; echo "host: $HOST"; echo "cores: $CORES ram_gb: $RAM_GB"
  lscpu 2>/dev/null | grep -E "Model name|Thread|Core|Socket|L1d|L2|L3|MHz"
  gcc --version | head -1; mpirun --version 2>/dev/null | head -1; echo "flags: $FLAGS"
} > "$OUT/entorno.txt"

TOTAL_S=0; NCONF=0
for alg in $ALGS; do
  disponible "$alg" || { echo "[omitido] $alg (falta fuente o compilador)"; continue; }
  for n in $NS; do
    if es_serial "$alg"; then plist=1; else plist=$PS; fi
    for p in $plist; do
      reps=$(reps_para "$n")
      est=$(awk -v r="$reps" -v n="$n" -v g="$(rate_gflops "$alg")" -v p="$p" \
        'BEGIN{printf "%d", r*2*n^3/(g*1e9*(p^0.85))}')
      TOTAL_S=$((TOTAL_S + est)); NCONF=$((NCONF+1))
      key="${alg}_${n}_${p}"
      grep -qx "$key" "$HECHOS" && continue
      [ "$DRY" = 1 ] && { echo "  $key reps=$reps ~${est}s"; continue; }

      if ! compilar "$alg" "$n"; then
        echo "$key compilacion" >> "$FALLOS"; echo "[ERROR compilar] $key"; continue; fi
      exe=$BIN/${alg}_${n}
      if [ "$DESDE_ARCHIVO" = 1 ]; then
        asegurar_matrices "$n" || { echo "$key matrices" >> "$FALLOS"; echo "[ERROR matrices] $key"; continue; }
        ARGS_ARCHIVO="--archivo-a $DIR_MATRICES/A_$n.txt --archivo-b $DIR_MATRICES/B_$n.txt"
      fi
      if es_serial "$alg"; then
        runner=""
      else
        runner="mpirun --bind-to core --map-by core -np $p"
      fi
      echo "[$(date +%H:%M:%S)] $key reps=$reps (est ~${est}s)"
      [ "$DESDE_ARCHIVO" = 1 ] && enfriar_cache "$n"
      tmp=$(mktemp)
      # Page faults y OOM: /usr/bin/time (GNU) suma los fallos de todos los
      # procesos hijos (los ranks MPI); el contador oom_kill del cgroup detecta
      # si el kernel mato algo durante la corrida.
      medidor=""; [ -x /usr/bin/time ] && medidor="/usr/bin/time -o $tmp.time -f %e,%R,%F,%M"
      oom0=$(awk '/^oom_kill /{print $2}' /sys/fs/cgroup/memory.events 2>/dev/null); oom0=${oom0:-0}
      OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
        timeout "$TIMEOUT" $medidor $runner "$exe" --csv --repeticiones "$reps" $ARGS_ARCHIVO > "$tmp" 2>"$tmp.err"
      rc=$?
      oom1=$(awk '/^oom_kill /{print $2}' /sys/fs/cgroup/memory.events 2>/dev/null); oom1=${oom1:-0}
      if [ $rc -eq 0 ]; then estado=ok; elif [ $rc -eq 124 ]; then estado=timeout
      elif [ "$oom1" -gt "$oom0" ]; then estado=oom_kill; else estado=error_rc$rc; fi
      # pared_s, fallos menores, fallos mayores, max RSS (KB, de un solo proceso)
      tiempos=$(tail -1 "$tmp.time" 2>/dev/null); tiempos=${tiempos:-,,,}
      [ -f "$OUT/estado.csv" ] || echo "perfil,n,procesos,algoritmo,estado,pared_s,fallos_menores,fallos_mayores,maxrss_kb" > "$OUT/estado.csv"
      echo "$PERFIL,$n,$p,$alg,$estado,$tiempos" >> "$OUT/estado.csv"
      [ "$estado" = ok ] || echo "  -> $estado"
      rm -f "$tmp.time"
      if [ $rc -ne 0 ]; then
        echo "$key rc=$rc $estado $(head -c 200 "$tmp.err" | tr '\n' ' ')" >> "$FALLOS"
        echo "  [FALLO rc=$rc] ver $FALLOS"
        # Un OOM o timeout no se reintenta al reanudar (podria volver a tumbar la sesion).
        case $estado in oom_kill|timeout) echo "$key" >> "$HECHOS";; esac
      else
        # Normaliza: serial = n,m,p,alg,rep,seg,gf,gen ; mpi = n,m,p,procs,alg,rep,seg,gf,gen
        filas=$(awk -F, -v perfil="$PERFIL" -v alg="$alg" -v p="$p" 'NR>1 && NF>=8 {
          if (NF==8) printf "%s,%s,%s,%s,%s,%s,%s,%s\n", perfil,$1,1,alg,$5,$6,$7,$8
          else       printf "%s,%s,%s,%s,%s,%s,%s,%s\n", perfil,$1,$4,alg,$6,$7,$8,$9 }' "$tmp" | tee -a "$CSV" | wc -l)
        if [ "$filas" -gt 0 ]; then
          echo "$key" >> "$HECHOS"
        else
          echo "  [FALLO] $key: la salida no tiene filas validas (binario viejo?), ver $FALLOS"
          echo "$key sin_filas $(head -c 200 "$tmp" | tr '\n' ' ')" >> "$FALLOS"
        fi
      fi
      rm -f "$tmp" "$tmp.err"
    done
  done
done
echo "Configuraciones: $NCONF  tiempo estimado total: ~$((TOTAL_S/3600))h $((TOTAL_S%3600/60))m (cota gruesa)"
[ "$DRY" = 1 ] || echo "Listo. Datos en $CSV (fallos: $FALLOS)"

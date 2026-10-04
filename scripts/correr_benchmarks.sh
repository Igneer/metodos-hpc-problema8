#!/usr/bin/env bash
# Corre todas las implementaciones variando N (matrices cuadradas) y cantidad
# de procesos, y deja UN csv crudo para procesar fuera del servidor.
# Solo usa bash/awk/gcc/mpicc: no requiere Python.
#
# Uso:
#   scripts/correr_benchmarks.sh --perfil laptop            # 4 cores / 8 GB
#   scripts/correr_benchmarks.sh --perfil servidor          # autodetecta cores y RAM
#   scripts/correr_benchmarks.sh --perfil servidor --dry-run   # solo muestra el plan
# Opciones: --cores C  --ram-gb G  --ns "1024 2048"  --ps "1 2 4"
#           --algoritmos "serial_ikj mpi_ikj"  --timeout SEG  --salida DIR
#           --limite-ram [--cgroup SWAP_GB]   experimento de cruce del limite de RAM
#           --nmax-serial N   N maximo comun a seriales/mpi_ijk/mpi_ikj (defecto 20000)
#           --solo seriales|mpi   corre solo ese grupo (defecto: todos)
#           --repeticiones R  repeticiones para N > --n-chico (defecto 5)
#           --reps-chicas R --n-chico N   repeticiones para N <= N (defecto 10 y 2000)
#
# Salida: resultados/<perfil>_<cores>c/crudo.csv
#   perfil,n,procesos,algoritmo,repeticion,segundos,gflops,segundos_generacion
#   (segundos = solo el producto [+ distribucion/recoleccion en MPI]; segundos_generacion =
#    tiempo de crear A y B, medido una vez por corrida, repetido en cada fila)
# Es reanudable: si se corta, volver a lanzar el mismo comando salta lo hecho.
# Recomendado en el servidor:  nohup scripts/correr_benchmarks.sh ... &
set -uo pipefail
cd "$(dirname "$0")/.."

PERFIL=servidor; CORES=""; RAM_GB=""; NS=""; PS=""; ALGS=""; TIMEOUT=""
SOLO=""; LIMITE=0; SWAP_GB=""; NMAX_SERIAL=20000; REPS=5; REPS_CHICAS=10; N_CHICO=2000
SALIDA=""; DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --perfil) PERFIL=$2; shift 2;;   --cores) CORES=$2; shift 2;;
    --ram-gb) RAM_GB=$2; shift 2;;   --ns) NS=$2; shift 2;;
    --ps) PS=$2; shift 2;;           --algoritmos) ALGS=$2; shift 2;;
    --timeout) TIMEOUT=$2; shift 2;; --salida) SALIDA=$2; shift 2;;
    --limite-ram) LIMITE=1; shift;;  --cgroup) SWAP_GB=$2; shift 2;;
    --nmax-serial) NMAX_SERIAL=$2; shift 2;;
    --solo) SOLO=$2; shift 2;;
    --repeticiones) REPS=$2; shift 2;;  --reps-chicas) REPS_CHICAS=$2; shift 2;;
    --n-chico) N_CHICO=$2; shift 2;;
    --dry-run) DRY=1; shift;;
    *) echo "Argumento desconocido: $1" >&2; exit 1;;
  esac
done

# ---------- Hardware ----------
# En contenedores (JupyterHub/Kubernetes) nproc y /proc/meminfo muestran el HOST;
# el limite real esta en los cgroups (v2). Se leen primero; --cores/--ram-gb mandan.
if [ -z "$CORES" ] && [ -r /sys/fs/cgroup/cpu.max ]; then
  CORES=$(awk '$1!="max"{c=int(($1+$2-1)/$2); print (c<1?1:c)}' /sys/fs/cgroup/cpu.max)
fi
if [ -z "$RAM_GB" ] && [ -r /sys/fs/cgroup/memory.max ]; then
  RAM_GB=$(awk '$1!="max"{printf "%d", $1/1073741824}' /sys/fs/cgroup/memory.max)
fi
[ -z "$CORES" ] && { [ "$PERFIL" = laptop ] && CORES=4 || CORES=$(nproc); }
if [ -z "$RAM_GB" ]; then
  RAM_GB=$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo 2>/dev/null)
  RAM_GB=${RAM_GB:-8}
  [ "$PERFIL" = laptop ] && [ "$RAM_GB" -gt 8 ] && RAM_GB=8
fi
# Limite teorico: se descarta una configuracion solo si su memoria estimada supera la RAM
# del perfil (sin margen; pasarse mata el proceso por OOM, no hay swap).
PRESUPUESTO=$(awk -v g="$RAM_GB" 'BEGIN{printf "%d", g*1073741824}')

[ -z "$NS" ] && NS="500 1000 1500 2000 3000 4000 5000 7500 10000"
# Procesos MPI: 1 y potencias de 2 hasta CORES, mas CORES (p=1 sirve de base del
# speedup MPI, aparte del serial). Nunca oversubscribe.
if [ -z "$PS" ]; then
  PS=1; p=2; while [ $p -lt "$CORES" ]; do PS="$PS $p"; p=$((p*2)); done
  [ "$CORES" -gt 1 ] && PS="$PS $CORES"
fi
ALL="serial_ikj serial_ikj_bloqueado mpi_ikj mpi_ikj_bloqueado ref_pdgemm"   # sin ijk (disponible con --algoritmos)
case "$SOLO" in
  seriales) [ -z "$ALGS" ] && ALGS="serial_ikj serial_ikj_bloqueado";;
  mpi)      [ -z "$ALGS" ] && ALGS="mpi_ikj mpi_ikj_bloqueado ref_pdgemm";;
  ""|todos) ;;
  *) echo "--solo debe ser seriales, mpi o todos" >&2; exit 1;;
esac
[ -z "$ALGS" ] && { [ "$LIMITE" = 1 ] && ALGS=${ALL//serial_ijk/} || ALGS=$ALL; }
if [ "$LIMITE" = 1 ]; then ALGS=${ALGS//mpi_ijk/}; TIMEOUT=${TIMEOUT:-900}; fi
TIMEOUT=${TIMEOUT:-21600}   # 6 h por combinacion (todas las repeticiones)

# N maximo COMUN para los seriales (y mpi_ijk, mpi_ikj): lo fija el mas lento.
# Con --nmax-serial N (defecto 20000). Los demas MPI no tienen tope propio.
nmax_alg() { case $1 in serial_*|mpi_ijk|mpi_ikj) echo "$NMAX_SERIAL";; *) echo 999999;; esac; }
# Bytes de memoria residente aprox. = coef * N^2 (todo corre en un solo nodo).
coef_mem() { # alg p
  case $1 in
    serial_*) echo 24;;
    mpi_ijk|mpi_ikj|mpi_ikj_bloqueado) echo $((8 * ($2 + 4)));;   # B replicada en cada rank
    ref_pdgemm) echo 48;;   # A,B,C globales en la raiz + bloques locales (~2x)
    *)                   echo 64;;                  # (no usado)
  esac
}
es_serial() { case $1 in serial_*) return 0;; *) return 1;; esac; }
reps_para() { [ "$1" -le "$N_CHICO" ] && echo "$REPS_CHICAS" || echo "$REPS"; }   # igual para todos los algoritmos
rate_gflops() { case $1 in *ijk) echo 1;; *ikj) echo 6;; *) echo 25;; esac; }

OUT=${SALIDA:-resultados/${PERFIL}_${CORES}c}
[ "$LIMITE" = 1 ] && OUT=${SALIDA:-resultados/limite_ram_${RAM_GB}g_${CORES}c}
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

# ---------- Plan ----------
HOST=$(hostname)
echo "Perfil=$PERFIL cores=$CORES RAM=${RAM_GB}GB limite_teorico=$((PRESUPUESTO/1073741824))GB"
echo "N: $NS"; echo "P: $PS"; echo "Salida: $CSV"
{ echo "fecha: $(date -Is)"; echo "host: $HOST"; echo "cores: $CORES ram_gb: $RAM_GB"
  lscpu 2>/dev/null | grep -E "Model name|Thread|Core|Socket|L1d|L2|L3|MHz"
  gcc --version | head -1; mpirun --version 2>/dev/null | head -1; echo "flags: $FLAGS"
} > "$OUT/entorno.txt"

# ---------- Modo --limite-ram ----------
# Barre N alrededor del limite de RAM (footprint/RAM = 0.5 .. 1.5) con 1
# repeticion por punto, p en {1, CORES}, sin ijk. Cada fila registra el estado
# (ok / timeout / killed_rcNN) porque pasado el limite el proceso puede morir
# por OOM en lugar de volverse lento (depende de si hay swap).
# --ram-gb es la RAM NOMINAL a cruzar. --cgroup SWAP_GB (requiere root) fija
# MemoryMax=RAM_GB y MemorySwapMax=SWAP_GB, asi el limite es reproducible y
# un exceso no afecta al resto de la maquina.
if [ "$LIMITE" = 1 ]; then
  FRACS=${FRACS:-"0.5 0.8 1.0 1.2 1.5"}
  LCSV=$OUT/limite_ram.csv
  [ -f "$LCSV" ] || echo "perfil,n,procesos,algoritmo,ratio_ram,estado,segundos_programa,gflops,segundos_pared" > "$LCSV"
  LIM_PS=""; [ "$CORES" -gt 1 ] && LIM_PS="$CORES"
  LIMITADOR=""
  if [ -n "$SWAP_GB" ]; then
    LIMITADOR="systemd-run --scope --quiet -p MemoryMax=${RAM_GB}G -p MemorySwapMax=${SWAP_GB}G"
  fi
  echo "Modo limite-ram: RAM nominal=${RAM_GB}GB fracciones=[$FRACS] timeout=${TIMEOUT}s"
  echo "Swap del sistema:"; swapon --show 2>/dev/null || echo "  (no se pudo consultar)"
  [ -z "$LIMITADOR" ] && echo "AVISO: sin --cgroup el limite es la RAM real de la maquina."
  for alg in $ALGS; do
    disponible "$alg" || { echo "[omitido] $alg"; continue; }
    if es_serial "$alg"; then plist=1; else plist=$LIM_PS; fi
    for p in $plist; do
      c=$(coef_mem "$alg" "$p")
      for f in $FRACS; do
        n=$(awk -v f="$f" -v r="$RAM_GB" -v c="$c" 'BEGIN{printf "%d", int(sqrt(f*r*1073741824/c)/500+0.5)*500}')
        ratio=$(awk -v n="$n" -v r="$RAM_GB" -v c="$c" 'BEGIN{printf "%.2f", c*n*n/(r*1073741824)}')
        key="lim_${alg}_${n}_${p}"
        grep -qx "$key" "$HECHOS" && continue
        [ "$DRY" = 1 ] && { echo "  $key ratio=$ratio"; continue; }
        compilar "$alg" "$n" || { echo "$key compilacion" >> "$FALLOS"; continue; }
        exe=$BIN/${alg}_${n}
        thr=1; runner=""
        es_serial "$alg" || runner="mpirun ${MPIRUN_EXTRA:-} --bind-to core --map-by core -np $p"
        echo "[$(date +%H:%M:%S)] $key ratio_ram=$ratio"
        tmp=$(mktemp); t0=$(date +%s.%N)
        OPENBLAS_NUM_THREADS=$thr OMP_NUM_THREADS=$thr \
          timeout "$TIMEOUT" $LIMITADOR $runner "$exe" --csv --repeticiones 1 > "$tmp" 2>"$tmp.err"
        rc=$?; pared=$(awk -v a="$t0" -v b="$(date +%s.%N)" 'BEGIN{printf "%.2f", b-a}')
        seg=""; gf=""
        if [ $rc -eq 0 ]; then
          estado=ok
          read -r seg gf < <(awk -F, 'NR==2{ if (NF==8) print $6,$7; else print $7,$8 }' "$tmp")
        elif [ $rc -eq 124 ]; then estado=timeout
        else estado=killed_rc$rc; fi
        echo "$PERFIL,$n,$p,$alg,$ratio,$estado,${seg:-},${gf:-},$pared" >> "$LCSV"
        echo "  -> $estado (${pared}s pared)"
        [ $rc -eq 0 ] || echo "$key rc=$rc $(head -c 200 "$tmp.err" | tr '\n' ' ')" >> "$FALLOS"
        echo "$key" >> "$HECHOS"; rm -f "$tmp" "$tmp.err"
      done
    done
  done
  [ "$DRY" = 1 ] || echo "Listo. Datos en $LCSV"
  exit 0
fi

TOTAL_S=0; NCONF=0
for alg in $ALGS; do
  disponible "$alg" || { echo "[omitido] $alg (falta fuente o compilador)"; continue; }
  for n in $NS; do
    [ "$n" -gt "$(nmax_alg "$alg")" ] && continue
    if es_serial "$alg"; then plist=1; else plist=$PS; fi
    for p in $plist; do
      mem=$(( $(coef_mem "$alg" "$p") * n * n ))
      # No se descarta nada por memoria: si excede la RAM se corre igual y se
      # registra el efecto (page faults, OOM) en estado.csv.
      ratio=$(awk -v m="$mem" -v r="$RAM_GB" 'BEGIN{printf "%.2f", m/(r*1073741824)}')
      excede=""; [ "$mem" -gt "$PRESUPUESTO" ] && excede=" [EXCEDE RAM x$ratio]"
      reps=$(reps_para "$n")
      est=$(awk -v r="$reps" -v n="$n" -v g="$(rate_gflops "$alg")" -v p="$p" \
        'BEGIN{printf "%d", r*2*n^3/(g*1e9*(p^0.85))}')
      TOTAL_S=$((TOTAL_S + est)); NCONF=$((NCONF+1))
      key="${alg}_${n}_${p}"
      # Hecho = esta en hechos.txt Y (tiene filas en crudo.csv O termino mal,
      # timeout/oom, que no deja filas). Si hechos.txt dice ok pero crudo.csv
      # no tiene datos (p.ej. se borro/reemplazo el csv), se vuelve a correr.
      if grep -qx "$key" "$HECHOS"; then
        if awk -F, -v a="$alg" -v n="$n" -v p="$p" '$2==n && $3==p && $4==a {f=1; exit} END{exit !f}' "$CSV"; then
          continue
        fi
        if [ -f "$OUT/estado.csv" ] && awk -F, -v a="$alg" -v n="$n" -v p="$p" \
             '$2==n && $3==p && $4==a {e=$6} END{exit !(e!="" && e!="ok")}' "$OUT/estado.csv"; then
          continue
        fi
        echo "[reintento] $key figura como hecho pero no tiene datos en $CSV"
      fi
      [ "$DRY" = 1 ] && { echo "  $key reps=$reps ~${est}s mem~$((mem/1048576))MB$excede"; continue; }

      if ! compilar "$alg" "$n"; then
        echo "$key compilacion" >> "$FALLOS"; echo "[ERROR compilar] $key"; continue; fi
      exe=$BIN/${alg}_${n}
      if es_serial "$alg"; then
        runner=""; thr=1
      else
        runner="mpirun ${MPIRUN_EXTRA:-} --bind-to core --map-by core -np $p"; thr=1
      fi
      echo "[$(date +%H:%M:%S)] $key reps=$reps (est ~${est}s)$excede"
      tmp=$(mktemp)
      # Page faults y OOM: /usr/bin/time (GNU) suma los fallos de todos los
      # procesos hijos (los ranks MPI); el contador oom_kill del cgroup detecta
      # si el kernel mato algo durante la corrida.
      medidor=""; [ -x /usr/bin/time ] && medidor="/usr/bin/time -o $tmp.time -f %e,%R,%F,%M"
      oom0=$(awk '/^oom_kill /{print $2}' /sys/fs/cgroup/memory.events 2>/dev/null); oom0=${oom0:-0}
      OPENBLAS_NUM_THREADS=$thr OMP_NUM_THREADS=$thr \
        timeout "$TIMEOUT" $medidor $runner "$exe" --csv --repeticiones "$reps" > "$tmp" 2>"$tmp.err"
      rc=$?
      oom1=$(awk '/^oom_kill /{print $2}' /sys/fs/cgroup/memory.events 2>/dev/null); oom1=${oom1:-0}
      if [ $rc -eq 0 ]; then estado=ok; elif [ $rc -eq 124 ]; then estado=timeout
      elif [ "$oom1" -gt "$oom0" ]; then estado=oom_kill; else estado=error_rc$rc; fi
      # pared_s, fallos menores, fallos mayores, max RSS (KB, de un solo proceso)
      tiempos=$(tail -1 "$tmp.time" 2>/dev/null); tiempos=${tiempos:-,,,}
      [ -f "$OUT/estado.csv" ] || echo "perfil,n,procesos,algoritmo,ratio_ram,estado,pared_s,fallos_menores,fallos_mayores,maxrss_kb" > "$OUT/estado.csv"
      echo "$PERFIL,$n,$p,$alg,$ratio,$estado,$tiempos" >> "$OUT/estado.csv"
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

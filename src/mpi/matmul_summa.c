/*
 * matmul_summa.c (MPI)
 *
 * C = A * B (A: N x M, B: M x P, C: N x P) con distribucion 2D por
 * bloques (SUMMA, van de Geijn & Watts 1995): los procesos se acomodan
 * en una grilla Pr x Pc (`size` no necesita ser cuadrado perfecto: se
 * factoriza en el par mas parecido a un cuadrado posible), cada uno
 * guarda un bloque de A, un bloque de B y acumula su bloque de C. A
 * diferencia de la version row-wise (matmul_ijk.c/matmul_ikj.c),
 * ningun proceso necesita B completa.
 *
 * El problema de la grilla no cuadrada: A se distribuye N-sobre-Pr /
 * M-sobre-Pc, B se distribuye M-sobre-Pr / P-sobre-Pc. El eje M queda
 * partido dos veces distintas -en Pc bloques para A, en Pr bloques
 * para B- y esos cortes no coinciden si Pr != Pc. La solucion:
 * fusionar ambas listas de cortes en una sola secuencia de tramos,
 * cada uno intacto dentro de un unico bloque de A y de un unico
 * bloque de B, y usar esa fusion como los pasos del loop principal.
 *
 * Asimetria A vs B en cada paso: el tramo de A es un rango de
 * *columnas* dentro del bloque local (no contiguo en memoria: hay que
 * empaquetarlo en un buffer aparte). El tramo de B es un rango de
 * *filas* enteras dentro del bloque local (ya contiguo tal cual esta
 * guardado): el dueno lo transmite apuntando directo a esa porcion,
 * sin copiar.
 *
 * N, M, P son constantes de compilacion (config.h) y todo es estatico
 * (BSS, sin malloc/free), igual que las otras versiones MPI.
 *
 * Uso:
 *   mpirun -np <procesos> ./matmul_summa [--semilla S] [--repeticiones R] [--csv]
 */

#include <mpi.h>
#include <stdio.h>
#include <string.h>

#include "../common/cli.h"
#include "../common/config.h"
#include "../common/matrix.h"
#include "../common/particiones.h"

/* Cota de tramos tras fusionar los cortes de A (hasta Pc) y B (hasta
 * Pr): a lo sumo Pc+Pr-1 tramos distintos, mas el cierre en M. */
#define MAX_SEGMENTOS (2 * MAX_PROCESOS + 2)

static real_t A[(size_t)N * M];
static real_t B[(size_t)M * P];
static real_t C[(size_t)N * P];

static real_t A_local[(size_t)N * M];
static real_t B_local[(size_t)M * P];
static real_t C_local[(size_t)N * P];

static real_t panel_A[(size_t)N * M];
static real_t panel_B[(size_t)M * P];

static real_t scratch_A[(size_t)N * M];
static real_t scratch_B[(size_t)M * P];
static real_t scratch_C[(size_t)N * P];

static int bloques_N[MAX_PROCESOS], inicio_N[MAX_PROCESOS];
static int bloques_M_A[MAX_PROCESOS], inicio_M_A[MAX_PROCESOS]; /* M partido en Pc bloques (columnas de A) */
static int bloques_M_B[MAX_PROCESOS], inicio_M_B[MAX_PROCESOS]; /* M partido en Pr bloques (filas de B) */
static int bloques_P[MAX_PROCESOS], inicio_P[MAX_PROCESOS];

/* Tramos fusionados: cada uno vive dentro de un unico bloque de A
 * (columna de grilla `dueno_col`) y de un unico bloque de B (fila de
 * grilla `dueno_fila`), con su offset local dentro de esos bloques. */
static int seg_ancho[MAX_SEGMENTOS];
static int seg_dueno_col[MAX_SEGMENTOS];
static int seg_dueno_fila[MAX_SEGMENTOS];
static int seg_offset_A[MAX_SEGMENTOS];
static int seg_offset_B[MAX_SEGMENTOS];
static int num_segmentos;

enum { TAG_A = 1, TAG_B = 2, TAG_C = 3 };

/* Copia un bloque `filas` x `columnas` desde `origen` (stride
 * `stride_origen`) a `destino` (stride `stride_destino`). Sirve para
 * empaquetar (destino contiguo) o desempaquetar (origen contiguo) un
 * sub-bloque de una matriz mas grande guardada por filas. */
static void copiar_bloque(const real_t *origen, size_t stride_origen,
                           real_t *destino, size_t stride_destino,
                           size_t filas, size_t columnas) {
    for (size_t i = 0; i < filas; i++) {
        memcpy(destino + i * stride_destino, origen + i * stride_origen,
               columnas * sizeof(real_t));
    }
}

static void multiplicar_acumulando(const real_t *restrict a, const real_t *restrict b,
                                    real_t *restrict c, int n, int m, int p) {
    for (int i = 0; i < n; i++) {
        real_t *restrict fila_c = c + (size_t)i * p;
        for (int k = 0; k < m; k++) {
            const real_t a_ik = a[(size_t)i * m + k];
            const real_t *restrict fila_b = b + (size_t)k * p;
            for (int j = 0; j < p; j++) {
                fila_c[j] += a_ik * fila_b[j];
            }
        }
    }
}

/* Factoriza `p` en Pr x Pc lo mas parecido a un cuadrado posible
 * (Pr <= Pc, Pr el divisor de p mas grande con Pr*Pr <= p). Con p
 * primo degenera a 1 x p (sigue siendo correcto, solo deja de tener
 * ventaja de memoria sobre la version 1D). */
static void factorizar_grilla(int p, int *pr_out, int *pc_out) {
    int mejor = 1;
    for (int cand = 1; (size_t)cand * (size_t)cand <= (size_t)p; cand++) {
        if (p % cand == 0) mejor = cand;
    }
    *pr_out = mejor;
    *pc_out = p / mejor;
}

/* Devuelve el indice del bloque (0-based) al que pertenece
 * `indice_global`, dado un arreglo de inicios de bloque ordenado
 * ascendente. Es la operacion inversa de calcular_particiones. */
static int ubicar_bloque(const int *inicios, int cantidad, int indice_global) {
    int resultado = 0;
    for (int i = 0; i < cantidad; i++) {
        if (inicios[i] <= indice_global) resultado = i;
        else break;
    }
    return resultado;
}

/* Fusiona dos listas de inicios de bloque ya ordenadas (`a` de
 * tamanio na, `b` de tamanio nb) en una unica lista de cortes
 * distintos y crecientes, terminada en `total` (el largo del eje que
 * ambas particionan). Cada tramo [cortes[i], cortes[i+1]) queda
 * intacto dentro de un bloque de `a` y uno de `b` a la vez, sin
 * importar que `a` y `b` tengan distinta cantidad de bloques. Devuelve
 * la cantidad de cortes escritos en `out`. */
static int fusionar_cortes(const int *a, int na, const int *b, int nb,
                            int total, int *out) {
    int i = 0, j = 0, n = 0, ultimo = -1;
    while (i < na || j < nb) {
        int candidato;
        if (j >= nb || (i < na && a[i] <= b[j])) {
            candidato = a[i++];
        } else {
            candidato = b[j++];
        }
        if (candidato != ultimo) {
            out[n++] = candidato;
            ultimo = candidato;
        }
    }
    if (ultimo != total) out[n++] = total;
    return n;
}

int main(int argc, char *argv[]) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size > MAX_PROCESOS) {
        if (rank == 0) {
            fprintf(stderr, "Error: %d procesos pedidos, MAX_PROCESOS=%d\n",
                    size, MAX_PROCESOS);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int Pr, Pc;
    factorizar_grilla(size, &Pr, &Pc);
    int fila = rank / Pc;
    int col = rank % Pc;

    char info_tamanio[64];
    snprintf(info_tamanio, sizeof(info_tamanio), "N=%d M=%d P=%d", N, M, P);

    ConfiguracionBenchmark config;
    if (!cli_parsear_argumentos(argc, argv, &config, info_tamanio)) {
        MPI_Finalize();
        return 1;
    }

    if (calcular_particiones((size_t)N, 1, Pr, MAX_PROCESOS, bloques_N, inicio_N) != PARTICIONES_EXITO ||
        calcular_particiones((size_t)M, 1, Pc, MAX_PROCESOS, bloques_M_A, inicio_M_A) != PARTICIONES_EXITO ||
        calcular_particiones((size_t)M, 1, Pr, MAX_PROCESOS, bloques_M_B, inicio_M_B) != PARTICIONES_EXITO ||
        calcular_particiones((size_t)P, 1, Pc, MAX_PROCESOS, bloques_P, inicio_P) != PARTICIONES_EXITO) {
        if (rank == 0) fprintf(stderr, "Error al calcular las particiones 2D.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    size_t n_loc = (size_t)bloques_N[fila];
    size_t m_loc_A = (size_t)bloques_M_A[col];
    size_t m_loc_B = (size_t)bloques_M_B[fila];
    size_t p_loc = (size_t)bloques_P[col];

    /* Fusion de cortes de M (una sola vez, no depende de los datos ni
     * cambia entre repeticiones) y precalculo de cada tramo: quien lo
     * transmite en A, quien en B, y su offset dentro del bloque local
     * de cada uno. */
    int cortes[MAX_SEGMENTOS];
    int num_cortes = fusionar_cortes(inicio_M_A, Pc, inicio_M_B, Pr, M, cortes);
    num_segmentos = num_cortes - 1;
    for (int s = 0; s < num_segmentos; s++) {
        int m0 = cortes[s], m1 = cortes[s + 1];
        int dc = ubicar_bloque(inicio_M_A, Pc, m0);
        int df = ubicar_bloque(inicio_M_B, Pr, m0);
        seg_ancho[s] = m1 - m0;
        seg_dueno_col[s] = dc;
        seg_dueno_fila[s] = df;
        seg_offset_A[s] = m0 - inicio_M_A[dc];
        seg_offset_B[s] = m0 - inicio_M_B[df];
    }

    /* row_comm agrupa procesos de la misma fila de grilla, ordenados
     * por columna (key=col): la posicion de cada rank dentro de
     * row_comm coincide con su columna, asi `root=dueno_col` en el
     * Bcast apunta al proceso correcto. col_comm es el espejo. */
    MPI_Comm row_comm, col_comm;
    MPI_Comm_split(MPI_COMM_WORLD, fila, col, &row_comm);
    MPI_Comm_split(MPI_COMM_WORLD, col, fila, &col_comm);

    if (rank == 0) {
        if (config.archivo_a) {
            if (matriz_leer_archivo(A, N, M, config.archivo_a) != 0) {
                fprintf(stderr, "Error leyendo %s\n", config.archivo_a);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        } else {
            matriz_llenar_aleatoria(A, N, M, config.semilla);
        }
        if (config.archivo_b) {
            if (matriz_leer_archivo(B, M, P, config.archivo_b) != 0) {
                fprintf(stderr, "Error leyendo %s\n", config.archivo_b);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        } else {
            matriz_llenar_aleatoria(B, M, P, config.semilla + 1);
        }
    }

    double flops = 2.0 * (double)N * (double)M * (double)P;

    if (rank == 0) {
        if (config.csv) {
            printf("n,m,p,procesos,algoritmo,repeticion,segundos,gflops\n");
        } else {
            printf("N=%d M=%d P=%d procesos=%d (grilla %dx%d, %d tramos) "
                   "algoritmo=summa repeticiones=%d\n",
                   N, M, P, size, Pr, Pc, num_segmentos, config.repeticiones);
        }
    }

    for (int r = 0; r < config.repeticiones; r++) {
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();

        /* Distribucion 2D: rank 0 empaqueta el bloque de cada proceso
         * (no contiguo en la matriz completa) y lo manda en un solo
         * mensaje; A usa la particion de M en Pc bloques, B la de Pr. */
        if (rank == 0) {
            for (int fr = 0; fr < Pr; fr++) {
                for (int fc = 0; fc < Pc; fc++) {
                    int dest = fr * Pc + fc;
                    size_t nl = (size_t)bloques_N[fr], mlA = (size_t)bloques_M_A[fc];
                    size_t mlB = (size_t)bloques_M_B[fr], pl = (size_t)bloques_P[fc];

                    const real_t *origen_A = A + (size_t)inicio_N[fr] * M + (size_t)inicio_M_A[fc];
                    const real_t *origen_B = B + (size_t)inicio_M_B[fr] * P + (size_t)inicio_P[fc];

                    if (dest == 0) {
                        copiar_bloque(origen_A, M, A_local, mlA, nl, mlA);
                        copiar_bloque(origen_B, P, B_local, pl, mlB, pl);
                    } else {
                        copiar_bloque(origen_A, M, scratch_A, mlA, nl, mlA);
                        MPI_Send(scratch_A, (int)(nl * mlA), MPI_DOUBLE, dest, TAG_A, MPI_COMM_WORLD);
                        copiar_bloque(origen_B, P, scratch_B, pl, mlB, pl);
                        MPI_Send(scratch_B, (int)(mlB * pl), MPI_DOUBLE, dest, TAG_B, MPI_COMM_WORLD);
                    }
                }
            }
        } else {
            MPI_Recv(A_local, (int)(n_loc * m_loc_A), MPI_DOUBLE, 0, TAG_A,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(B_local, (int)(m_loc_B * p_loc), MPI_DOUBLE, 0, TAG_B,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        matriz_poner_cero(C_local, n_loc, p_loc);

        /* Loop de SUMMA: un paso por tramo fusionado (no por bloque:
         * un tramo puede ser una porcion parcial del bloque de A y/o
         * de B del proceso que lo posee). */
        for (int s = 0; s < num_segmentos; s++) {
            size_t ancho = (size_t)seg_ancho[s];

            /* A: el tramo es un rango de columnas -> no contiguo en
             * A_local, el dueno lo empaqueta en panel_A antes de
             * transmitir. Todos (dueno incluido) usan panel_A. */
            if (col == seg_dueno_col[s]) {
                copiar_bloque(A_local + seg_offset_A[s], m_loc_A, panel_A, ancho, n_loc, ancho);
            }
            MPI_Bcast(panel_A, (int)(n_loc * ancho), MPI_DOUBLE, seg_dueno_col[s], row_comm);

            /* B: el tramo es un rango de filas enteras -> ya
             * contiguo en B_local, el dueno transmite apuntando
             * directo ahi, sin copiar. */
            real_t *buf_B = (fila == seg_dueno_fila[s])
                                 ? B_local + (size_t)seg_offset_B[s] * p_loc
                                 : panel_B;
            MPI_Bcast(buf_B, (int)(ancho * p_loc), MPI_DOUBLE, seg_dueno_fila[s], col_comm);

            multiplicar_acumulando(panel_A, buf_B, C_local, (int)n_loc, (int)ancho, (int)p_loc);
        }

        /* Recoleccion de C: cada proceso manda su bloque (ya
         * contiguo) a rank 0, que lo escribe en su posicion dentro
         * de C. */
        if (rank == 0) {
            copiar_bloque(C_local, p_loc, C, P, n_loc, p_loc);
            for (int fr = 0; fr < Pr; fr++) {
                for (int fc = 0; fc < Pc; fc++) {
                    int origen_rank = fr * Pc + fc;
                    if (origen_rank == 0) continue;
                    size_t nl = (size_t)bloques_N[fr], pl = (size_t)bloques_P[fc];
                    MPI_Recv(scratch_C, (int)(nl * pl), MPI_DOUBLE, origen_rank, TAG_C,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    real_t *destino = C + (size_t)inicio_N[fr] * P + (size_t)inicio_P[fc];
                    copiar_bloque(scratch_C, pl, destino, P, nl, pl);
                }
            }
        } else {
            MPI_Send(C_local, (int)(n_loc * p_loc), MPI_DOUBLE, 0, TAG_C, MPI_COMM_WORLD);
        }

        double t1 = MPI_Wtime();
        double transcurrido = t1 - t0;

        if (rank == 0) {
            double gflops = flops / transcurrido / 1e9;
            if (config.csv) {
                printf("%d,%d,%d,%d,summa,%d,%.9f,%.4f\n",
                       N, M, P, size, r, transcurrido, gflops);
            } else {
                printf("repeticion %d: %.6f s (%.3f GFLOPS)\n", r, transcurrido, gflops);
            }
            if (config.imprimir) {
                matriz_imprimir(C, N, P);
            }
        }
    }

    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    MPI_Finalize();
    return 0;
}

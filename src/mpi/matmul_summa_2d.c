/*
 * matmul_summa_2d.c (MPI)
 *
 * C = A * B con SUMMA 2D modular: particion (etapa 1), topologia cartesiana,
 * distribucion/reconstruccion y kernel (etapa 2) y coreografia por
 * segmentos (etapa 3). N, M y P son constantes de compilacion (config.h)
 * y las matrices son estaticas row-major.
 *
 * Uso:
 *   mpirun -np <procesos> ./matmul_summa_2d [--semilla S] [--repeticiones R] [--csv]
 *          [--archivo-a RUTA] [--archivo-b RUTA] [--imprimir]
 */

#include <mpi.h>
#include <stdio.h>

#include "../common/cli.h"
#include "../common/config.h"
#include "../common/matrix.h"
#include "2d/kernel_propio.h"
#include "2d/particion_2d.h"
#include "2d/summa_2d.h"
#include "2d/topologia_2d.h"
#include "2d/transferencias_2d.h"

static real_t A[(size_t)N * M];
static real_t B[(size_t)M * P];
static real_t C[(size_t)N * P];

/* Capacidad conservadora: admite cualquier grilla en runtime. */
static real_t A_local[(size_t)N * M];
static real_t B_local[(size_t)M * P];
static real_t C_local[(size_t)N * P];
static real_t panel_A[(size_t)N * M];
static real_t panel_B[(size_t)M * P];

static Particion2D particion;

static void abortar(const char *mensaje) {
    fprintf(stderr, "Error: %s\n", mensaje);
    MPI_Abort(MPI_COMM_WORLD, 1);
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size > MAX_PROCESOS) abortar("mas procesos que MAX_PROCESOS");

    char info_tamanio[64];
    snprintf(info_tamanio, sizeof(info_tamanio), "N=%d M=%d P=%d", N, M, P);
    ConfiguracionBenchmark config;
    if (!cli_parsear_argumentos(argc, argv, &config, info_tamanio)) {
        MPI_Finalize();
        return 1;
    }

    Topologia2D topologia;
    if (topologia_2d_crear(MPI_COMM_WORLD, &topologia)) abortar("no se pudo crear la grilla");
    if (calcular_particion_2d(topologia.filas_grilla, topologia.columnas_grilla,
                              &particion)) {
        abortar("no se pudo calcular la particion 2D");
    }
    /* Todo lo colectivo ocurre en la grilla (reorder=1 puede cambiar ranks). */
    int soy_raiz = topologia.fila_actual == 0 && topologia.columna_actual == 0;

    if (soy_raiz) {
        if (config.archivo_a) {
            if (matriz_leer_archivo(A, N, M, config.archivo_a) != 0) abortar("leyendo archivo A");
        } else {
            matriz_llenar_aleatoria(A, N, M, config.semilla);
        }
        if (config.archivo_b) {
            if (matriz_leer_archivo(B, M, P, config.archivo_b) != 0) abortar("leyendo archivo B");
        } else {
            matriz_llenar_aleatoria(B, M, P, config.semilla + 1);
        }
    }

    double flops = 2.0 * (double)N * (double)M * (double)P;
    if (soy_raiz) {
        if (config.csv) {
            printf("n,m,p,procesos,algoritmo,repeticion,segundos,gflops\n");
        } else {
            printf("N=%d M=%d P=%d procesos=%d (grilla %dx%d, %d tramos) "
                   "algoritmo=summa_2d repeticiones=%d\n",
                   N, M, P, size, topologia.filas_grilla, topologia.columnas_grilla,
                   particion.num_segmentos, config.repeticiones);
        }
    }

    for (int r = 0; r < config.repeticiones; r++) {
        MPI_Barrier(topologia.grilla);
        double t0 = MPI_Wtime();
        int fallo = distribuir_2d(&topologia, &particion, MATRIZ_2D_A, A, A_local) ||
                    distribuir_2d(&topologia, &particion, MATRIZ_2D_B, B, B_local);
        double t1 = MPI_Wtime();
        fallo = fallo || summa_2d(&topologia, &particion, kernel_propio_ikj,
                                  A_local, B_local, C_local, panel_A, panel_B);
        double t2 = MPI_Wtime();
        fallo = fallo || reconstruir_2d(&topologia, &particion, MATRIZ_2D_C, C_local, C);
        double t3 = MPI_Wtime();
        if (fallo) abortar("fallo la distribucion, SUMMA o la recoleccion");

        /* Tiempos: el maximo entre procesos representa a la corrida. */
        double local[4] = {t3 - t0, t1 - t0, t2 - t1, t3 - t2}, maximo[4];
        MPI_Reduce(local, maximo, 4, MPI_DOUBLE, MPI_MAX, topologia.rank_raiz, topologia.grilla);

        if (soy_raiz) {
            double gflops = flops / maximo[0] / 1e9;
            if (config.csv) {
                printf("%d,%d,%d,%d,summa_2d,%d,%.9f,%.4f\n",
                       N, M, P, size, r, maximo[0], gflops);
            } else {
                printf("repeticion %d: %.6f s (%.3f GFLOPS) "
                       "[distribucion %.6f, SUMMA %.6f, recoleccion %.6f]\n",
                       r, maximo[0], gflops, maximo[1], maximo[2], maximo[3]);
            }
            if (config.imprimir) matriz_imprimir(C, N, P);
        }
    }

    topologia_2d_liberar(&topologia);
    MPI_Finalize();
    return 0;
}

#include <mpi.h>
#include <stdio.h>

#include "../common/cli.h"
#include "../common/config.h"
#include "../common/matrix.h"
#include "../common/particiones.h"

static real_t A[(size_t)N * M];
static real_t B[(size_t)M * P];
static real_t C[(size_t)N * P];

/* Peor caso: np=1 recibe las N filas enteras. */
static real_t local_A[(size_t)N * M];
static real_t local_C[(size_t)N * P];

static int sendcounts[MAX_PROCESOS];
static int displs[MAX_PROCESOS];
static int recvcounts_C[MAX_PROCESOS];
static int displs_C[MAX_PROCESOS];

static void multiplicar_ijk(const real_t *restrict a, const real_t *restrict b,
                             real_t *restrict c, int n, int m, int p) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            real_t acumulador = 0.0;
            for (int k = 0; k < m; k++) {
                acumulador += a[(size_t)i * m + k] * b[(size_t)k * p + j];
            }
            c[(size_t)i * p + j] = acumulador;
        }
    }
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

    char info_tamanio[64];
    snprintf(info_tamanio, sizeof(info_tamanio), "N=%d M=%d P=%d", N, M, P);

    /* mpirun propaga el mismo argv a todos los ranks, asi que todos
     * parsean igual (sin Bcast de config). Con argumentos invalidos,
     * cada rank imprime su propio mensaje de uso. */
    ConfiguracionBenchmark config;
    if (!cli_parsear_argumentos(argc, argv, &config, info_tamanio)) {
        MPI_Finalize();
        return 1;
    }

    int estado = calcular_particiones((size_t)N, (size_t)M, size, MAX_PROCESOS,
                                       sendcounts, displs);
    if (estado != PARTICIONES_EXITO) {
        if (rank == 0) fprintf(stderr, "Error al calcular las particiones de A.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* C tiene P columnas, no M: si P != M la particion cambia. */
    estado = calcular_particiones((size_t)N, (size_t)P, size, MAX_PROCESOS,
                                   recvcounts_C, displs_C);
    if (estado != PARTICIONES_EXITO) {
        if (rank == 0) fprintf(stderr, "Error al calcular las particiones de C.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int mis_elementos_A = sendcounts[rank];
    size_t mis_filas = (size_t)mis_elementos_A / (size_t)M;
    int mis_elementos_C = (int)(mis_filas * (size_t)P);

    double t_gen0 = MPI_Wtime();
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
    double segundos_generacion = MPI_Wtime() - t_gen0;

    double flops = 2.0 * (double)N * (double)M * (double)P;

    if (rank == 0) {
        if (config.csv) {
            printf("n,m,p,procesos,algoritmo,repeticion,segundos,gflops,segundos_generacion\n");
        } else {
            printf("N=%d M=%d P=%d procesos=%d algoritmo=ijk_mpi repeticiones=%d\n",
                   N, M, P, size, config.repeticiones);
        }
    }

    for (int r = 0; r < config.repeticiones; r++) {
        if (mis_elementos_C > 0) {
            matriz_poner_cero(local_C, mis_filas, P);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();

        MPI_Bcast(B, M * P, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatterv(A, sendcounts, displs, MPI_DOUBLE,
                     local_A, mis_elementos_A, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        multiplicar_ijk(local_A, B, local_C, (int)mis_filas, M, P);

        MPI_Gatherv(local_C, mis_elementos_C, MPI_DOUBLE,
                    C, recvcounts_C, displs_C, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        double t1 = MPI_Wtime();
        double transcurrido = t1 - t0;

        if (rank == 0) {
            double gflops = flops / transcurrido / 1e9;
            if (config.csv) {
                printf("%d,%d,%d,%d,ijk_mpi,%d,%.9f,%.4f,%.9f\n",
                       N, M, P, size, r, transcurrido, gflops, segundos_generacion);
            } else {
                printf("repeticion %d: %.6f s (%.3f GFLOPS)\n", r, transcurrido, gflops);
            }
            if (config.imprimir) {
                matriz_imprimir(C, N, P);
            }
        }
    }

    MPI_Finalize();
    return 0;
}

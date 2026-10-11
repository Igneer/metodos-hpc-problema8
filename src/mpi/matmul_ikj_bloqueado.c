#include <mpi.h>
#include <stdio.h>

#include "../common/cli.h"
#include "../common/config.h"
#include "../common/kernels.h"
#include "../common/matrix.h"
#include "../common/mpi_comun.h"
#include "../common/bloque.h"

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

int main(int argc, char *argv[]) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

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

    const TamanioBloque tb = bloque_calcular();

    mpi_preparar_particiones(rank, size, sendcounts, displs, recvcounts_C, displs_C);

    int mis_elementos_A = sendcounts[rank];
    size_t mis_filas = (size_t)mis_elementos_A / (size_t)M;
    int mis_elementos_C = (int)(mis_filas * (size_t)P);

    double t_gen0 = MPI_Wtime();
    if (rank == 0 && cli_cargar_matrices(&config, A, B, N, M, P) != 0) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    double segundos_generacion = MPI_Wtime() - t_gen0;

    if (rank == 0) {
        cli_imprimir_encabezado(&config, "ikj_bloqueado_mpi", NULL, size, N, M, P);
    }

    for (int r = 0; r < config.repeticiones; r++) {
        /* ikj bloqueado acumula con += sobre local_C: hay que ponerla en cero
         * antes de cada repeticion. */
        if (mis_elementos_C > 0) {
            matriz_poner_cero(local_C, mis_filas, P);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();

        MPI_Bcast(B, M * P, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatterv(A, sendcounts, displs, MPI_DOUBLE,
                     local_A, mis_elementos_A, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        multiplicar_ikj_bloqueada(local_A, B, local_C, (int)mis_filas, M, P, tb);

        MPI_Gatherv(local_C, mis_elementos_C, MPI_DOUBLE,
                    C, recvcounts_C, displs_C, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        double t1 = MPI_Wtime();

        if (rank == 0) {
            cli_imprimir_repeticion(&config, "ikj_bloqueado_mpi", size, N, M, P, r,
                                    t1 - t0, segundos_generacion);
            if (config.imprimir) {
                matriz_imprimir(C, N, P);
            }
        }
    }

    MPI_Finalize();
    return 0;
}

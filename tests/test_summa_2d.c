/* Prueba de la etapa 3: SUMMA 2D contra un producto serial de referencia.
 * Usa valores enteros pequenios, de modo que la comparacion es exacta.
 * Compilar y ejecutar (ver tests/probar_summa_2d.sh):
 *   mpicc -std=c11 -Wall -Wextra -DN=5 -DM=6 -DP=7 tests/test_summa_2d.c \
 *     src/mpi/2d/<todos los .c> src/common/particiones.c -o /tmp/test_summa_2d
 *   mpirun --oversubscribe -np 6 /tmp/test_summa_2d */
#include <mpi.h>
#include <stdio.h>

#include "../src/common/config.h"
#include "../src/mpi/2d/kernel_propio.h"
#include "../src/mpi/2d/particion_2d.h"
#include "../src/mpi/2d/summa_2d.h"
#include "../src/mpi/2d/topologia_2d.h"
#include "../src/mpi/2d/transferencias_2d.h"

static real_t A[(size_t)N * M], B[(size_t)M * P], C[(size_t)N * P], ref[(size_t)N * P];
static real_t A_loc[(size_t)N * M], B_loc[(size_t)M * P], C_loc[(size_t)N * P];
static real_t panel_A[(size_t)N * M], panel_B[(size_t)M * P];
static Particion2D particion;

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, procesos, errores = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &procesos);

    Topologia2D t;
    if (topologia_2d_crear(MPI_COMM_WORLD, &t) ||
        calcular_particion_2d(t.filas_grilla, t.columnas_grilla, &particion)) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int raiz = rank == t.rank_raiz;
    if (raiz) {
        for (size_t i = 0; i < N; i++) for (size_t j = 0; j < M; j++) A[i * M + j] = (real_t)((i * 3 + j * 5) % 7) - 3;
        for (size_t i = 0; i < M; i++) for (size_t j = 0; j < P; j++) B[i * P + j] = (real_t)((i * 2 + j * 7) % 5) - 2;
        for (size_t i = 0; i < N; i++)
            for (size_t k = 0; k < M; k++)
                for (size_t j = 0; j < P; j++) ref[i * P + j] += A[i * M + k] * B[k * P + j];
    }

    /* Dos corridas seguidas: C_local debe reiniciarse en cada SUMMA. */
    for (int repeticion = 0; repeticion < 2; repeticion++) {
        errores += distribuir_2d(&t, &particion, MATRIZ_2D_A, A, A_loc);
        errores += distribuir_2d(&t, &particion, MATRIZ_2D_B, B, B_loc);
        errores += summa_2d(&t, &particion, kernel_propio_ikj, A_loc, B_loc, C_loc, panel_A, panel_B);
        errores += reconstruir_2d(&t, &particion, MATRIZ_2D_C, C_loc, C);
        if (raiz)
            for (size_t i = 0; i < (size_t)N * P; i++)
                if (C[i] != ref[i]) errores++;
    }

    topologia_2d_liberar(&t);
    int total = 0;
    MPI_Reduce(&errores, &total, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        if (total == 0) printf("SUMMA 2D: pruebas OK (np=%d, N=%d M=%d P=%d)\n", procesos, N, M, P);
        else printf("SUMMA 2D: %d fallos (np=%d, N=%d M=%d P=%d)\n", total, procesos, N, M, P);
    }
    MPI_Finalize();
    return total != 0;
}

/* Prueba de la etapa 2: topologia, datatypes, distribucion/reconstruccion y
 * kernel con strides. N, M y P se pueden cambiar con -DN/-DM/-DP.
 * Compilar (desde la raiz) y ejecutar:
 *   mpicc -std=c11 -Wall -Wextra -DN=5 -DM=6 -DP=7 tests/test_infraestructura_2d.c \
 *     src/mpi/2d/<todos los .c> src/common/particiones.c -o /tmp/test_infra_2d
 *   mpirun --oversubscribe -np 6 /tmp/test_infra_2d */
#include <mpi.h>
#include <stdio.h>

#include "../src/common/config.h"
#include "../src/mpi/2d/kernel_propio.h"
#include "../src/mpi/2d/particion_2d.h"
#include "../src/mpi/2d/topologia_2d.h"
#include "../src/mpi/2d/transferencias_2d.h"

static real_t A[(size_t)N * M], B[(size_t)M * P], C[(size_t)N * P];
static real_t A_rec[(size_t)N * M], B_rec[(size_t)M * P], C_rec[(size_t)N * P];
static real_t A_loc[(size_t)N * M], B_loc[(size_t)M * P], C_loc[(size_t)N * P];
static Particion2D particion;

/* Valor distinto para cada posicion y cada matriz. */
static real_t valor(int matriz, size_t i, size_t j, size_t columnas) {
    return (real_t)(matriz * 1000 + i * columnas + j + 1);
}

static int errores = 0;
static void verificar(int condicion, const char *mensaje) {
    if (!condicion) {
        fprintf(stderr, "FALLO: %s\n", mensaje);
        errores++;
    }
}

/* El bloque local debe contener los valores globales de su region. */
static int bloque_correcto(const real_t *local, const Bloque2D *b, int matriz,
                           size_t columnas_globales) {
    for (size_t i = 0; i < b->filas; i++)
        for (size_t j = 0; j < b->columnas; j++)
            if (local[i * b->columnas + j] !=
                valor(matriz, b->inicio_fila + i, b->inicio_columna + j,
                      columnas_globales))
                return 0;
    return 1;
}

static void probar_kernel(void) {
    /* A 3x2 con stride 4, B 2x3 con stride 5, C 3x3 con stride 6, C previo = 1. */
    real_t a[3 * 4], b[2 * 5], c[3 * 6], esperado[3][3];
    for (int i = 0; i < 12; i++) a[i] = i + 1;
    for (int i = 0; i < 10; i++) b[i] = 2 * i - 3;
    for (int i = 0; i < 18; i++) c[i] = 1;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            esperado[i][j] = 1;
            for (int k = 0; k < 2; k++) esperado[i][j] += a[i * 4 + k] * b[k * 5 + j];
        }
    kernel_propio_ikj(a, b, c, 3, 2, 3, 4, 5, 6);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            verificar(c[i * 6 + j] == esperado[i][j], "kernel con strides");
    /* Las columnas de relleno de C no se tocan. */
    verificar(c[3] == 1 && c[4] == 1 && c[5] == 1, "kernel respeta el relleno de C");
    /* Dimension cero: no accede a los buffers. */
    kernel_propio_ikj(NULL, NULL, NULL, 0, 2, 3, 1, 1, 1);
    kernel_propio_ikj(NULL, NULL, NULL, 3, 0, 3, 1, 1, 1);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, procesos;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &procesos);

    Topologia2D t;
    verificar(topologia_2d_crear(MPI_COMM_WORLD, &t) == 0, "crear topologia");
    verificar(t.filas_grilla * t.columnas_grilla == procesos, "grilla cubre los procesos");

    /* Coordenadas <-> rank y subcomunicadores coherentes. */
    int coords[2] = {t.fila_actual, t.columna_actual}, rank_grilla, rank_cart, tam;
    MPI_Comm_rank(t.grilla, &rank_grilla);
    MPI_Cart_rank(t.grilla, coords, &rank_cart);
    verificar(rank_cart == rank_grilla, "Cart_rank inverso de Cart_coords");
    MPI_Comm_size(t.fila, &tam);
    verificar(tam == t.columnas_grilla, "tamanio del grupo de fila");
    MPI_Comm_rank(t.fila, &rank_cart);
    verificar(rank_cart == t.columna_actual, "rank en fila = columna de grilla");
    MPI_Comm_size(t.columna, &tam);
    verificar(tam == t.filas_grilla, "tamanio del grupo de columna");
    MPI_Comm_rank(t.columna, &rank_cart);
    verificar(rank_cart == t.fila_actual, "rank en columna = fila de grilla");

    verificar(calcular_particion_2d(t.filas_grilla, t.columnas_grilla, &particion) == 0,
              "calcular particion 2d");

    if (rank == t.rank_raiz) {
        for (size_t i = 0; i < N; i++) for (size_t j = 0; j < M; j++) A[i * M + j] = valor(0, i, j, M);
        for (size_t i = 0; i < M; i++) for (size_t j = 0; j < P; j++) B[i * P + j] = valor(1, i, j, P);
        for (size_t i = 0; i < N; i++) for (size_t j = 0; j < P; j++) C[i * P + j] = valor(2, i, j, P);
        probar_kernel();
    }

    BloquesProceso2D mios;
    verificar(obtener_bloques_2d(&particion, t.fila_actual, t.columna_actual, &mios) == 0,
              "bloques propios");

    /* Distribucion: cada proceso recibe exactamente su region. */
    verificar(distribuir_2d(&t, &particion, MATRIZ_2D_A, A, A_loc) == 0, "distribuir A");
    verificar(distribuir_2d(&t, &particion, MATRIZ_2D_B, B, B_loc) == 0, "distribuir B");
    verificar(distribuir_2d(&t, &particion, MATRIZ_2D_C, C, C_loc) == 0, "distribuir C");
    verificar(bloque_correcto(A_loc, &mios.a, 0, M), "bloque local de A");
    verificar(bloque_correcto(B_loc, &mios.b, 1, P), "bloque local de B");
    verificar(bloque_correcto(C_loc, &mios.c, 2, P), "bloque local de C");

    /* Ida y vuelta: lo reconstruido debe igualar el original. */
    verificar(reconstruir_2d(&t, &particion, MATRIZ_2D_A, A_loc, A_rec) == 0, "reconstruir A");
    verificar(reconstruir_2d(&t, &particion, MATRIZ_2D_B, B_loc, B_rec) == 0, "reconstruir B");
    verificar(reconstruir_2d(&t, &particion, MATRIZ_2D_C, C_loc, C_rec) == 0, "reconstruir C");
    if (rank == t.rank_raiz) {
        for (size_t i = 0; i < (size_t)N * M; i++) verificar(A[i] == A_rec[i], "A ida y vuelta");
        for (size_t i = 0; i < (size_t)M * P; i++) verificar(B[i] == B_rec[i], "B ida y vuelta");
        for (size_t i = 0; i < (size_t)N * P; i++) verificar(C[i] == C_rec[i], "C ida y vuelta");
    }

    /* Un bloque vacio usa count 0 con un datatype basico. */
    Bloque2D vacio = {0, 3, 0, 0};
    TipoBloque2D tv;
    verificar(tipo_bloque_2d_crear(&vacio, N, M, &tv) == 0 && tv.count == 0 &&
              tv.tipo == MPI_DOUBLE, "bloque vacio");
    tipo_bloque_2d_liberar(&tv);

    topologia_2d_liberar(&t);
    verificar(t.grilla == MPI_COMM_NULL && t.fila == MPI_COMM_NULL &&
              t.columna == MPI_COMM_NULL, "comunicadores liberados");

    int total = 0;
    MPI_Reduce(&errores, &total, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        if (total == 0)
            printf("Infraestructura 2D: pruebas OK (np=%d, N=%d M=%d P=%d)\n", procesos, N, M, P);
        else
            printf("Infraestructura 2D: %d fallos\n", total);
    }
    MPI_Finalize();
    return total != 0;
}

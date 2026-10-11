#include <mpi.h>
#include <stdio.h>
#include <string.h>

#include "../common/cli.h"
#include "../common/config.h"
#include "../common/matrix.h"

#ifndef NB
#define NB 64
#endif

/* Interfaz C de BLACS y Fortran de ScaLAPACK (la biblioteca no trae header). */
extern void Cblacs_pinfo(int *rank, int *size);
extern void Cblacs_get(int contexto, int que, int *valor);
extern void Cblacs_gridinit(int *contexto, char *orden, int filas, int columnas);
extern void Cblacs_gridinfo(int contexto, int *filas, int *columnas, int *fila, int *columna);
extern void Cblacs_gridexit(int contexto);
extern void Cblacs_exit(int no_finalizar_mpi);
extern int numroc_(int *n, int *nb, int *iproc, int *isrcproc, int *nprocs);
extern void descinit_(int *desc, int *m, int *n, int *mb, int *nb, int *irsrc,
                      int *icsrc, int *ictxt, int *lld, int *info);
extern void pdgemr2d_(int *m, int *n, double *a, int *ia, int *ja, int *desca,
                      double *b, int *ib, int *jb, int *descb, int *ictxt);
extern void pdgemm_(char *transa, char *transb, int *m, int *n, int *k,
                    double *alpha, double *a, int *ia, int *ja, int *desca,
                    double *b, int *ib, int *jb, int *descb, double *beta,
                    double *c, int *ic, int *jc, int *descc);

/* Matrices globales (solo la raiz las usa) y bloques locales. Cada bloque
 * local tiene a lo sumo tantas filas y columnas como la matriz global. */
static real_t A[(size_t)N * M];
static real_t B[(size_t)M * P];
static real_t C[(size_t)N * P];
static real_t A_local[(size_t)N * M];
static real_t B_local[(size_t)M * P];
static real_t C_local[(size_t)N * P];

static void abortar(const char *mensaje) {
    fprintf(stderr, "Error: %s\n", mensaje);
    MPI_Abort(MPI_COMM_WORLD, 1);
}

/* Descriptor de una matriz en la grilla 2D con bloques ciclicos NB x NB.
 * Devuelve el tamanio local en filas/columnas (numroc) y lld. */
static void describir_distribuida(int *desc, int filas, int columnas, int ctx,
                                  int fila_proc, int col_proc, int pr, int pc,
                                  int *filas_loc, int *cols_loc) {
    int nb = NB, cero = 0, info;
    *filas_loc = numroc_(&filas, &nb, &fila_proc, &cero, &pr);
    *cols_loc = numroc_(&columnas, &nb, &col_proc, &cero, &pc);
    int lld = *filas_loc > 1 ? *filas_loc : 1;
    descinit_(desc, &filas, &columnas, &nb, &nb, &cero, &cero, &ctx, &lld, &info);
    if (info != 0) abortar("descinit fallo");
}

/* Descriptor de la matriz completa en la grilla 1x1 de la raiz. Quien no
 * pertenece a esa grilla marca el contexto como -1 (convencion de ScaLAPACK). */
static void describir_global(int *desc, int filas, int columnas, int ctx_raiz,
                             int soy_raiz) {
    if (!soy_raiz) {
        for (int i = 0; i < 9; i++) desc[i] = -1;
        return;
    }
    int cero = 0, info, lld = filas;
    descinit_(desc, &filas, &columnas, &filas, &columnas, &cero, &cero, &ctx_raiz,
              &lld, &info);
    if (info != 0) abortar("descinit global fallo");
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char info_tamanio[64];
    snprintf(info_tamanio, sizeof(info_tamanio), "N=%d M=%d P=%d", N, M, P);
    ConfiguracionBenchmark config;
    if (!cli_parsear_argumentos(argc, argv, &config, info_tamanio)) {
        MPI_Finalize();
        return 1;
    }

    /* Grilla 2D Pr x Pc elegida por MPI; BLACS la crea y dice mis coordenadas. */
    int dims[2] = {0, 0};
    MPI_Dims_create(size, 2, dims);
    int pr = dims[0], pc = dims[1];

    int rank_blacs, size_blacs;
    Cblacs_pinfo(&rank_blacs, &size_blacs);
    int ctx_grilla, ctx_raiz;
    Cblacs_get(-1, 0, &ctx_grilla);
    Cblacs_gridinit(&ctx_grilla, "Row", pr, pc);
    Cblacs_get(-1, 0, &ctx_raiz);
    Cblacs_gridinit(&ctx_raiz, "Row", 1, 1);
    int pr_g, pc_g, mi_fila, mi_col;
    Cblacs_gridinfo(ctx_grilla, &pr_g, &pc_g, &mi_fila, &mi_col);
    int soy_raiz = rank == 0;

    /* A^T (MxN), B^T (PxM), C^T (PxN) en column-major; ver cabecera. */
    int descA_g[9], descB_g[9], descC_g[9], descA[9], descB[9], descC[9];
    int filas_a, cols_a, filas_b, cols_b, filas_c, cols_c;
    describir_global(descA_g, M, N, ctx_raiz, soy_raiz);
    describir_global(descB_g, P, M, ctx_raiz, soy_raiz);
    describir_global(descC_g, P, N, ctx_raiz, soy_raiz);
    describir_distribuida(descA, M, N, ctx_grilla, mi_fila, mi_col, pr_g, pc_g, &filas_a, &cols_a);
    describir_distribuida(descB, P, M, ctx_grilla, mi_fila, mi_col, pr_g, pc_g, &filas_b, &cols_b);
    describir_distribuida(descC, P, N, ctx_grilla, mi_fila, mi_col, pr_g, pc_g, &filas_c, &cols_c);

    double t_gen0 = MPI_Wtime();
    if (soy_raiz && cli_cargar_matrices(&config, A, B, N, M, P) != 0) {
        abortar("leyendo archivos de entrada");
    }
    double segundos_generacion = MPI_Wtime() - t_gen0;

    double flops = 2.0 * (double)N * (double)M * (double)P;
    if (soy_raiz) {
        if (config.csv) {
            printf("n,m,p,procesos,algoritmo,repeticion,segundos,gflops,segundos_generacion\n");
        } else {
            printf("N=%d M=%d P=%d procesos=%d (grilla %dx%d, bloque %d) "
                   "algoritmo=pdgemm_2d repeticiones=%d\n",
                   N, M, P, size, pr_g, pc_g, NB, config.repeticiones);
        }
    }

    int uno = 1, n_a = M, n_b = P, n_c = P, m_n = N, m_m = M;
    int dim_m = P, dim_n = N, dim_k = M;
    double alfa = 1.0, beta = 0.0;

    for (int r = 0; r < config.repeticiones; r++) {
        /* C se ignora con beta=0, pero se limpia para no heredar basura. */
        memset(C_local, 0, sizeof(real_t) * (size_t)filas_c * (size_t)cols_c);

        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();
        pdgemr2d_(&n_a, &m_n, A, &uno, &uno, descA_g, A_local, &uno, &uno, descA, &ctx_grilla);
        pdgemr2d_(&n_b, &m_m, B, &uno, &uno, descB_g, B_local, &uno, &uno, descB, &ctx_grilla);
        double t1 = MPI_Wtime();
        pdgemm_("N", "N", &dim_m, &dim_n, &dim_k, &alfa, B_local, &uno, &uno, descB,
                A_local, &uno, &uno, descA, &beta, C_local, &uno, &uno, descC);
        double t2 = MPI_Wtime();
        pdgemr2d_(&n_c, &m_n, C_local, &uno, &uno, descC, C, &uno, &uno, descC_g, &ctx_grilla);
        double t3 = MPI_Wtime();

        double local[4] = {t3 - t0, t1 - t0, t2 - t1, t3 - t2}, maximo[4];
        MPI_Reduce(local, maximo, 4, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (soy_raiz) {
            double gflops = flops / maximo[0] / 1e9;
            if (config.csv) {
                printf("%d,%d,%d,%d,pdgemm_2d,%d,%.9f,%.4f,%.9f\n",
                       N, M, P, size, r, maximo[0], gflops, segundos_generacion);
            } else {
                printf("repeticion %d: %.6f s (%.3f GFLOPS) "
                       "[distribucion %.6f, PDGEMM %.6f, recoleccion %.6f]\n",
                       r, maximo[0], gflops, maximo[1], maximo[2], maximo[3]);
            }
            if (config.imprimir) matriz_imprimir(C, N, P);
        }
    }

    if (ctx_raiz >= 0) Cblacs_gridexit(ctx_raiz);  /* -1 fuera de la grilla 1x1 */
    Cblacs_gridexit(ctx_grilla);
    Cblacs_exit(1);  /* 1: MPI lo finaliza esta aplicacion */
    MPI_Finalize();
    return 0;
}

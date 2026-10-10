#include "../common/timing.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../common/cli.h"
#include "../common/bloque.h"
#include "../common/config.h"
#include "../common/matrix.h"

/* Matrices estaticas: tamanio fijo conocido en tiempo de compilacion,
 * reservadas en el segmento estatico del programa (no malloc/free,
 * no VLA en el stack). */
static real_t A[(size_t)N * M];
static real_t B[(size_t)M * P];
static real_t C[(size_t)N * P];


static int min_int(int a, int b) {
    return a < b ? a : b;
}

static void multiplicar_ikj_bloqueada(const real_t *restrict a,
                                       const real_t *restrict b,
                                       real_t *restrict c,
                                       int n, int m, int p,
                                       TamanioBloque tb) {
    const int bi = tb.bi, bk = tb.bk, bj = tb.bj;
    for (int ii = 0; ii < n; ii += bi) {
        int i_fin = min_int(ii + bi, n);
        for (int jj = 0; jj < p; jj += bj) {
            int j_fin = min_int(jj + bj, p);
            for (int kk = 0; kk < m; kk += bk) {
                int k_fin = min_int(kk + bk, m);

                /* Cuerpo ikj de siempre, acotado al bloque actual. */
                for (int i = ii; i < i_fin; i++) {
                    real_t *restrict fila_c = c + (size_t)i * p;
                    for (int k = kk; k < k_fin; k++) {
                        const real_t a_ik = a[(size_t)i * m + k];
                        const real_t *restrict fila_b = b + (size_t)k * p;
                        for (int j = jj; j < j_fin; j++) {
                            fila_c[j] += a_ik * fila_b[j];
                        }
                    }
                }
            }
        }
    }
}

int main(int argc, char **argv) {
    char info_tamanio[96];
    const TamanioBloque tb = bloque_calcular();
    snprintf(info_tamanio, sizeof(info_tamanio),
             "N=%d M=%d P=%d, BI=%d BK=%d BJ=%d", N, M, P, tb.bi, tb.bk, tb.bj);

    ConfiguracionBenchmark config;
    if (!cli_parsear_argumentos(argc, argv, &config, info_tamanio)) {
        return 1;
    }

    double t_gen0 = ahora_segundos();
    if (config.archivo_a) {
        if (matriz_leer_archivo(A, N, M, config.archivo_a) != 0) {
            fprintf(stderr, "Error leyendo %s\n", config.archivo_a);
            return 1;
        }
    } else {
        matriz_llenar_aleatoria(A, N, M, config.semilla);
    }
    if (config.archivo_b) {
        if (matriz_leer_archivo(B, M, P, config.archivo_b) != 0) {
            fprintf(stderr, "Error leyendo %s\n", config.archivo_b);
            return 1;
        }
    } else {
        matriz_llenar_aleatoria(B, M, P, config.semilla + 1);
    }
    double segundos_generacion = ahora_segundos() - t_gen0;

    double flops = 2.0 * (double)N * (double)M * (double)P;

    if (config.csv) {
        printf("n,m,p,algoritmo,repeticion,segundos,gflops,segundos_generacion\n");
    } else {
        printf("N=%d M=%d P=%d algoritmo=ikj_bloqueado (BI=%d BK=%d BJ=%d) repeticiones=%d\n",
               N, M, P, tb.bi, tb.bk, tb.bj, config.repeticiones);
    }


    for (int r = 0; r < config.repeticiones; r++) {
        matriz_poner_cero(C, N, P);
        double t0 = ahora_segundos();
        multiplicar_ikj_bloqueada(A, B, C, N, M, P, tb);
        double t1 = ahora_segundos();
        double transcurrido = t1 - t0;
        double gflops = flops / transcurrido / 1e9;

        if (config.csv) {
            printf("%d,%d,%d,ikj_bloqueado,%d,%.9f,%.4f,%.9f\n",
                   N, M, P, r, transcurrido, gflops, segundos_generacion);
        } else {
            printf("repeticion %d: %.6f s (%.3f GFLOPS)\n", r, transcurrido, gflops);
        }
        if (config.imprimir) {
            matriz_imprimir(C, N, P);
        }
    }

    return 0;
}

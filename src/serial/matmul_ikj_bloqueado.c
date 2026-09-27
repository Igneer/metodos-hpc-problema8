/*
 * matmul_ikj_bloqueado.c
 *
 * Multiplicacion secuencial C = A * B (A: N x M, B: M x P, C: N x P)
 * usando el orden de loops ikj con cache blocking/tiling sobre las
 * tres dimensiones (i, k, j). Mismo algoritmo O(n*m*p) y mismo
 * cuerpo interno que matmul_ikj.c; la diferencia es que en vez de
 * barrer cada dimension de punta a punta, se recorre en bloques de
 * tamanio BI/BK/BJ (ver src/common/config.h) para que los datos
 * activos de A, B y C quepan juntos en la cache (L2 tipicamente) y
 * se reutilicen antes de ser desalojados.
 *
 * Por que hace falta esto ademas de ikj:
 * ikj ya logra que el loop mas interno (en j) sea stride-1 sobre B y
 * C. Pero para cada fila i de A, ese loop en k recorre las M filas
 * completas de B (M*P elementos). Si M*P*sizeof(real_t) no entra en
 * cache, B se vuelve a traer entero desde RAM en cada una de las N
 * iteraciones de i, en vez de reutilizarse. El blocking evita esto
 * acotando cuantas filas de A y columnas de B/C se procesan juntas
 * antes de pasar al siguiente bloque, para que el bloque de B en uso
 * siga resonando en cache mientras se reutiliza.
 *
 * N, M, P (y BI, BK, BJ) son constantes de compilacion (ver
 * src/common/config.h), no parametros de linea de comandos: las
 * matrices son estaticas (viven en BSS, no en el heap). Para otro
 * tamanio o bloque, recompilar con -DN=... -DM=... -DP=...
 * -DBI=... -DBK=... -DBJ=...
 *
 * Uso:
 *   matmul_ikj_bloqueado [--semilla S] [--repeticiones R] [--csv]
 *
 * Ejemplos:
 *   ./matmul_ikj_bloqueado --repeticiones 10
 *   ./matmul_ikj_bloqueado --csv > resultados_ikj_bloqueado.csv
 */

#include "../common/timing.h" /* primero: define _POSIX_C_SOURCE antes de <stdio.h> */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../common/cli.h"
#include "../common/config.h"
#include "../common/matrix.h"

/* Matrices estaticas: tamanio fijo conocido en tiempo de compilacion,
 * reservadas en el segmento estatico del programa (no malloc/free,
 * no VLA en el stack). */
static real_t A[(size_t)N * M];
static real_t B[(size_t)M * P];
static real_t C[(size_t)N * P];

/* ---------------------------------------------------------------------
 * Devuelve el menor entre a y b (para acotar el ultimo bloque parcial
 * cuando la dimension no es multiplo exacto del tamanio de bloque).
 * ------------------------------------------------------------------- */
static int min_int(int a, int b) {
    return a < b ? a : b;
}

/* ---------------------------------------------------------------------
 * Multiplicacion ikj bloqueada (cache blocking/tiling).
 *
 * Los tres loops externos (ii, kk, jj) recorren la matriz en bloques
 * de BI x BK x BJ. El cuerpo interno es exactamente el mismo ikj de
 * siempre (ver matmul_ikj.c), pero acotado al rango [ii, ii+BI) x
 * [kk, kk+BK) x [jj, jj+BJ) en vez de barrer 0..n, 0..m, 0..p enteros.
 *
 * Orden elegido para los loops de bloque (ii, kk, jj): para un bloque
 * de filas de C fijo (ii) y un bloque de columnas fijo (jj), se
 * recorren todos los bloques kk antes de pasar al siguiente jj. Eso
 * hace que el bloque de C (BI x BJ) se mantenga en cache acumulando
 * mientras se suman los distintos aportes de cada bloque de k, en
 * vez de tener que releerlo/reescribirlo desde RAM por cada kk.
 *
 * Nota: c debe venir inicializada en cero (se usa como acumulador).
 * ------------------------------------------------------------------- */
static void multiplicar_ikj_bloqueada(const real_t *restrict a,
                                       const real_t *restrict b,
                                       real_t *restrict c,
                                       int n, int m, int p) {
    for (int ii = 0; ii < n; ii += BI) {
        int i_fin = min_int(ii + BI, n);
        for (int jj = 0; jj < p; jj += BJ) {
            int j_fin = min_int(jj + BJ, p);
            for (int kk = 0; kk < m; kk += BK) {
                int k_fin = min_int(kk + BK, m);

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
    snprintf(info_tamanio, sizeof(info_tamanio),
             "N=%d M=%d P=%d, BI=%d BK=%d BJ=%d", N, M, P, BI, BK, BJ);

    ConfiguracionBenchmark config;
    if (!cli_parsear_argumentos(argc, argv, &config, info_tamanio)) {
        return 1;
    }

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

    double flops = 2.0 * (double)N * (double)M * (double)P;

    if (config.csv) {
        printf("n,m,p,algoritmo,repeticion,segundos,gflops\n");
    } else {
        printf("N=%d M=%d P=%d algoritmo=ikj_bloqueado (BI=%d BK=%d BJ=%d) repeticiones=%d\n",
               N, M, P, BI, BK, BJ, config.repeticiones);
    }


    for (int r = 0; r < config.repeticiones; r++) {
        matriz_poner_cero(C, N, P);
        double t0 = ahora_segundos();
        multiplicar_ikj_bloqueada(A, B, C, N, M, P);
        double t1 = ahora_segundos();
        double transcurrido = t1 - t0;
        double gflops = flops / transcurrido / 1e9;

        if (config.csv) {
            printf("%d,%d,%d,ikj_bloqueado,%d,%.9f,%.4f\n",
                   N, M, P, r, transcurrido, gflops);
        } else {
            printf("repeticion %d: %.6f s (%.3f GFLOPS)\n", r, transcurrido, gflops);
        }
        if (config.imprimir) {
            matriz_imprimir(C, N, P);
        }
    }

    return 0;
}

/*
 * matmul_ijk.c
 *
 * Multiplicacion secuencial C = A * B (A: N x M, B: M x P, C: N x P)
 * usando el orden de loops ijk ("de libro"). Se deja como programa
 * aparte solo como punto de comparacion contra matmul_ikj.c: mismo
 * algoritmo O(n*m*p), mismos datos, distinto patron de acceso a
 * memoria.
 *
 * N, M, P son constantes de compilacion (ver src/common/config.h),
 * no parametros de linea de comandos: las matrices son estaticas
 * (viven en BSS, no en el heap), asi que el tamanio tiene que
 * conocerse al compilar. Para otro tamanio, recompilar con
 * -DN=... -DM=... -DP=... (el Makefile lo expone como SIZE=...).
 *
 * Uso:
 *   matmul_ijk [--semilla S] [--repeticiones R] [--csv]
 *
 * Ejemplos:
 *   ./matmul_ijk --repeticiones 10
 *   ./matmul_ijk --csv > resultados_ijk.csv
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
 * Multiplicacion ijk (orden "de libro"): para cada C[i][j] recorre k.
 *
 * Acceso a b: b[k*p + j] con k variando en el loop intermedio -> salta
 * de a `p` elementos (stride = p*sizeof(real_t)) en cada paso de k.
 * Eso es un acceso "por columnas" de una matriz almacenada por filas:
 * cada acceso cae probablemente en una linea de cache distinta -> un
 * cache miss casi por cada elemento de b leido. Comparar los tiempos
 * de este programa contra matmul_ikj.c es justamente la evidencia
 * empirica de por que el orden de loops importa para la localidad
 * de cache (ver el comentario largo en matmul_ikj.c).
 * ------------------------------------------------------------------- */
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

int main(int argc, char **argv) {
    char info_tamanio[64];
    snprintf(info_tamanio, sizeof(info_tamanio), "N=%d M=%d P=%d", N, M, P);

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
        printf("N=%d M=%d P=%d algoritmo=ijk repeticiones=%d\n",
               N, M, P, config.repeticiones);
    }

    /* Cada repeticion corre la multiplicacion completa y se imprime
     * como dato crudo (una linea por corrida). La agregacion
     * estadistica (mediana, desvio) sobre estas corridas es
     * responsabilidad de un script externo de analisis (fase de
     * diseno experimental), no de este binario. */
    for (int r = 0; r < config.repeticiones; r++) {
        matriz_poner_cero(C, N, P);
        double t0 = ahora_segundos();
        multiplicar_ijk(A, B, C, N, M, P);
        double t1 = ahora_segundos();
        double transcurrido = t1 - t0;
        double gflops = flops / transcurrido / 1e9;

        if (config.csv) {
            printf("%d,%d,%d,ijk,%d,%.9f,%.4f,%.9f\n",
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

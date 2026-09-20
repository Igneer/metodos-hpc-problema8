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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../common/config.h"
#include "../common/matrix.h"

/* Matrices estaticas: tamanio fijo conocido en tiempo de compilacion,
 * reservadas en el segmento estatico del programa (no malloc/free,
 * no VLA en el stack). */
static real_t A[(size_t)N * M];
static real_t B[(size_t)M * P];
static real_t C[(size_t)N * P];

/* ---------------------------------------------------------------------
 * Reloj de alta resolucion (monotonico, no afectado por ajustes de
 * hora del sistema). Se usa alrededor de cada multiplicacion completa.
 * ------------------------------------------------------------------- */
static double ahora_segundos(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

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

typedef struct {
    unsigned semilla;
    int repeticiones;
    int csv;
} Configuracion;

static void imprimir_uso(const char *programa) {
    fprintf(stderr,
        "Uso: %s [--semilla S] [--repeticiones R] [--csv]\n"
        "  (N=%d M=%d P=%d fijados en compilacion, ver src/common/config.h)\n",
        programa, N, M, P);
}

static int parsear_argumentos(int argc, char **argv, Configuracion *config) {
    config->semilla = 42;
    config->repeticiones = 5;
    config->csv = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--semilla") == 0 && i + 1 < argc) {
            config->semilla = (unsigned)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--repeticiones") == 0 && i + 1 < argc) {
            config->repeticiones = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--csv") == 0) {
            config->csv = 1;
        } else if (strcmp(argv[i], "--help") == 0) {
            imprimir_uso(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Argumento desconocido: %s\n", argv[i]);
            imprimir_uso(argv[0]);
            return 0;
        }
    }

    if (config->repeticiones <= 0) config->repeticiones = 1;
    return 1;
}

int main(int argc, char **argv) {
    Configuracion config;
    if (!parsear_argumentos(argc, argv, &config)) {
        return 1;
    }

    matriz_llenar_aleatoria(A, N, M, config.semilla);
    matriz_llenar_aleatoria(B, M, P, config.semilla + 1);

    double flops = 2.0 * (double)N * (double)M * (double)P;

    if (config.csv) {
        printf("n,m,p,algoritmo,repeticion,segundos,gflops\n");
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
            printf("%d,%d,%d,ijk,%d,%.9f,%.4f\n",
                   N, M, P, r, transcurrido, gflops);
        } else {
            printf("repeticion %d: %.6f s (%.3f GFLOPS)\n", r, transcurrido, gflops);
        }
    }

    return 0;
}

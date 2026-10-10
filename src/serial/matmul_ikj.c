#include "../common/timing.h" 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../common/cli.h"
#include "../common/config.h"
#include "../common/matrix.h"

/* Matrices estaticas: tamanio fijo conocido en tiempo de compilacion,
   reservadas en el segmento estatico del programa (no malloc/free,
   no VLA en el stack). */
static real_t A[(size_t)N * M];
static real_t B[(size_t)M * P];
static real_t C[(size_t)N * P];

static void multiplicar_ikj(const real_t *restrict a, const real_t *restrict b,
                             real_t *restrict c, int n, int m, int p) {
    for (int i = 0; i < n; i++) {
        real_t *restrict fila_c = c + (size_t)i * p;
        for (int k = 0; k < m; k++) {
            const real_t a_ik = a[(size_t)i * m + k];
            const real_t *restrict fila_b = b + (size_t)k * p;
            for (int j = 0; j < p; j++) {
                fila_c[j] += a_ik * fila_b[j];
            }
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
        printf("N=%d M=%d P=%d algoritmo=ikj repeticiones=%d\n",
               N, M, P, config.repeticiones);
    }

    for (int r = 0; r < config.repeticiones; r++) {
        matriz_poner_cero(C, N, P);
        double t0 = ahora_segundos();
        multiplicar_ikj(A, B, C, N, M, P);
        double t1 = ahora_segundos();
        double transcurrido = t1 - t0;
        double gflops = flops / transcurrido / 1e9;

        if (config.csv) {
            printf("%d,%d,%d,ikj,%d,%.9f,%.4f,%.9f\n",
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

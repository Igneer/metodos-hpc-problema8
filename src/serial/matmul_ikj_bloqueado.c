#include "../common/timing.h" /* primero: define _POSIX_C_SOURCE antes de <stdio.h> */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../common/cli.h"
#include "../common/config.h"
#include "../common/kernels.h"
#include "../common/matrix.h"
#include "../common/bloque.h"

/* Matrices estaticas: tamanio fijo conocido en tiempo de compilacion,
 * reservadas en el segmento estatico del programa (no malloc/free,
 * no VLA en el stack). */
static real_t A[(size_t)N * M];
static real_t B[(size_t)M * P];
static real_t C[(size_t)N * P];

int main(int argc, char **argv) {
    char info_tamanio[96];
    const TamanioBloque tb = bloque_calcular();
    snprintf(info_tamanio, sizeof(info_tamanio),
             "N=%d M=%d P=%d, BI=%d BK=%d BJ=%d", N, M, P, tb.bi, tb.bk, tb.bj);
    char detalle_bloque[48];
    snprintf(detalle_bloque, sizeof(detalle_bloque), "BI=%d BK=%d BJ=%d",
             tb.bi, tb.bk, tb.bj);

    ConfiguracionBenchmark config;
    if (!cli_parsear_argumentos(argc, argv, &config, info_tamanio)) {
        return 1;
    }

    double t_gen0 = ahora_segundos();
    if (cli_cargar_matrices(&config, A, B, N, M, P) != 0) {
        return 1;
    }
    double segundos_generacion = ahora_segundos() - t_gen0;

    cli_imprimir_encabezado(&config, "ikj_bloqueado", detalle_bloque, 0, N, M, P);

    for (int r = 0; r < config.repeticiones; r++) {
        matriz_poner_cero(C, N, P);
        double t0 = ahora_segundos();
        multiplicar_ikj_bloqueada(A, B, C, N, M, P, tb);
        double t1 = ahora_segundos();

        cli_imprimir_repeticion(&config, "ikj_bloqueado", 0, N, M, P, r,
                                t1 - t0, segundos_generacion);
        if (config.imprimir) {
            matriz_imprimir(C, N, P);
        }
    }

    return 0;
}

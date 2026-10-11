#include "cli.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void imprimir_uso(const char *programa, const char *info_tamanio) {
    fprintf(stderr,
        "Uso: %s [--semilla S] [--repeticiones R] [--csv]\n"
        "         [--archivo-a RUTA] [--archivo-b RUTA] [--imprimir]\n"
        "  (%s fijados en compilacion, ver src/common/config.h)\n",
        programa, info_tamanio);
}

int cli_parsear_argumentos(int argc, char **argv, ConfiguracionBenchmark *config,
                            const char *info_tamanio) {
    config->semilla = 42;
    config->repeticiones = 5;
    config->csv = 0;
    config->archivo_a = NULL;
    config->archivo_b = NULL;
    config->imprimir = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--semilla") == 0 && i + 1 < argc) {
            config->semilla = (unsigned)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--repeticiones") == 0 && i + 1 < argc) {
            config->repeticiones = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--csv") == 0) {
            config->csv = 1;
        } else if (strcmp(argv[i], "--archivo-a") == 0 && i + 1 < argc) {
            config->archivo_a = argv[++i];
        } else if (strcmp(argv[i], "--archivo-b") == 0 && i + 1 < argc) {
            config->archivo_b = argv[++i];
        } else if (strcmp(argv[i], "--imprimir") == 0) {
            config->imprimir = 1;
        } else if (strcmp(argv[i], "--help") == 0) {
            imprimir_uso(argv[0], info_tamanio);
            return 0;
        } else {
            fprintf(stderr, "Argumento desconocido: %s\n", argv[i]);
            imprimir_uso(argv[0], info_tamanio);
            return 0;
        }
    }

    if (config->repeticiones <= 0) config->repeticiones = 1;
    return 1;
}

int cli_cargar_matrices(const ConfiguracionBenchmark *config, real_t *a,
                        real_t *b, int n, int m, int p) {
    if (config->archivo_a) {
        if (matriz_leer_archivo(a, n, m, config->archivo_a) != 0) {
            fprintf(stderr, "Error leyendo %s\n", config->archivo_a);
            return 1;
        }
    } else {
        matriz_llenar_aleatoria(a, n, m, config->semilla);
    }
    if (config->archivo_b) {
        if (matriz_leer_archivo(b, m, p, config->archivo_b) != 0) {
            fprintf(stderr, "Error leyendo %s\n", config->archivo_b);
            return 1;
        }
    } else {
        matriz_llenar_aleatoria(b, m, p, config->semilla + 1);
    }
    return 0;
}

void cli_imprimir_encabezado(const ConfiguracionBenchmark *config,
                             const char *algoritmo, const char *detalle,
                             int procesos, int n, int m, int p) {
    if (config->csv) {
        printf(procesos > 0
                   ? "n,m,p,procesos,algoritmo,repeticion,segundos,gflops,segundos_generacion\n"
                   : "n,m,p,algoritmo,repeticion,segundos,gflops,segundos_generacion\n");
        return;
    }
    printf("N=%d M=%d P=%d", n, m, p);
    if (procesos > 0) printf(" procesos=%d", procesos);
    printf(" algoritmo=%s", algoritmo);
    if (detalle) printf(" (%s)", detalle);
    printf(" repeticiones=%d\n", config->repeticiones);
}

void cli_imprimir_repeticion(const ConfiguracionBenchmark *config,
                             const char *algoritmo, int procesos, int n, int m,
                             int p, int repeticion, double segundos,
                             double segundos_generacion) {
    double gflops = 2.0 * (double)n * (double)m * (double)p / segundos / 1e9;
    if (config->csv) {
        printf("%d,%d,%d,", n, m, p);
        if (procesos > 0) printf("%d,", procesos);
        printf("%s,%d,%.9f,%.4f,%.9f\n", algoritmo, repeticion, segundos,
               gflops, segundos_generacion);
    } else {
        printf("repeticion %d: %.6f s (%.3f GFLOPS)\n", repeticion, segundos,
               gflops);
    }
}

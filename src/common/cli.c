#include "cli.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void imprimir_uso(const char *programa, const char *info_tamanio) {
    fprintf(stderr,
        "Uso: %s [--semilla S] [--repeticiones R] [--csv]\n"
        "  (%s fijados en compilacion, ver src/common/config.h)\n",
        programa, info_tamanio);
}

int cli_parsear_argumentos(int argc, char **argv, ConfiguracionBenchmark *config,
                            const char *info_tamanio) {
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

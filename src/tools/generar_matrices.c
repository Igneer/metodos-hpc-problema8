/*
 * generar_matrices.c
 *
 * Escribe A_<n>.txt y B_<n>.txt (matrices cuadradas n x n) en un directorio,
 * con las mismas semillas que usan los benchmarks (42 y 43), de modo que
 * leerlas desde archivo da exactamente los mismos datos que generarlas.
 *
 * Uso: generar_matrices N DIRECTORIO [SEMILLA]
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>

#include "../common/matrix.h"

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Uso: %s N DIRECTORIO [SEMILLA]\n", argv[0]);
        return 1;
    }
    size_t n = (size_t)strtoul(argv[1], NULL, 10);
    unsigned semilla = argc > 3 ? (unsigned)strtoul(argv[3], NULL, 10) : 42;

    real_t *m = malloc(n * n * sizeof(real_t));
    if (!m) { fprintf(stderr, "Sin memoria\n"); return 1; }

    for (int w = 0; w < 2; w++) {
        char ruta[1024];
        snprintf(ruta, sizeof ruta, "%s/%c_%zu.txt", argv[2], w ? 'B' : 'A', n);
        matriz_llenar_aleatoria(m, n, n, semilla + (unsigned)w);
        if (matriz_exportar_archivo(m, n, n, ruta) != 0) {
            fprintf(stderr, "Error escribiendo %s\n", ruta);
            free(m);
            return 1;
        }
    }
    free(m);
    return 0;
}

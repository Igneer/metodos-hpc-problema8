#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>

void matriz_llenar_aleatoria(real_t *matriz, size_t filas, size_t columnas,
                              unsigned semilla) {
    srand(semilla);
    size_t cantidad = filas * columnas;
    for (size_t idx = 0; idx < cantidad; idx++) {
        matriz[idx] = (real_t)rand() / ((real_t)RAND_MAX + 1.0);
    }
}

void matriz_poner_cero(real_t *matriz, size_t filas, size_t columnas) {
    size_t cantidad = filas * columnas;
    for (size_t idx = 0; idx < cantidad; idx++) {
        matriz[idx] = 0.0;
    }
}

void matriz_imprimir(const real_t *matriz, size_t filas, size_t columnas) {
    for (size_t i = 0; i < filas; i++) {
        for (size_t j = 0; j < columnas; j++) {
            printf("%8.4f ", matriz[i * columnas + j]);
        }
        printf("\n");
    }
}

int matriz_exportar_archivo(const real_t *matriz, size_t filas, size_t columnas,
                             const char *ruta) {
    FILE *archivo = fopen(ruta, "w");
    if (!archivo) {
        return 1;
    }

    for (size_t i = 0; i < filas; i++) {
        for (size_t j = 0; j < columnas; j++) {
            if (fprintf(archivo, j + 1 < columnas ? "%.10g " : "%.10g\n",
                        matriz[i * columnas + j]) < 0) {
                fclose(archivo);
                return 1;
            }
        }
    }

    fclose(archivo);
    return 0;
}

int matriz_leer_archivo(real_t *matriz, size_t filas, size_t columnas,
                         const char *ruta) {
    FILE *archivo = fopen(ruta, "r");
    if (!archivo) {
        return 1;
    }

    size_t cantidad = filas * columnas;
    for (size_t idx = 0; idx < cantidad; idx++) {
        if (fscanf(archivo, "%lf", &matriz[idx]) != 1) {
            fclose(archivo);
            return 1;
        }
    }

    fclose(archivo);
    return 0;
}

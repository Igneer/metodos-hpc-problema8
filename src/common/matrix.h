#ifndef HPC_MATRIX_H
#define HPC_MATRIX_H

#include <stddef.h>

typedef double real_t;

void matriz_llenar_aleatoria(real_t *matriz, size_t filas, size_t columnas,
                              unsigned semilla);

void matriz_poner_cero(real_t *matriz, size_t filas, size_t columnas);


void matriz_imprimir(const real_t *matriz, size_t filas, size_t columnas);


int matriz_exportar_archivo(const real_t *matriz, size_t filas, size_t columnas,
                             const char *ruta);


int matriz_leer_archivo(real_t *matriz, size_t filas, size_t columnas,
                         const char *ruta);

#endif /* HPC_MATRIX_H */

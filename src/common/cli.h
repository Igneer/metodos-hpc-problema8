#ifndef HPC_CLI_H
#define HPC_CLI_H

#include "matrix.h"

typedef struct {
    unsigned semilla;
    int repeticiones;
    int csv;
    /* Si no son NULL, A/B se cargan desde estos archivos (formato de
     * matriz_leer_archivo) en vez de llenarse con matriz_llenar_aleatoria.
     * Pensado para verificar el calculo con una matriz conocida. */
    const char *archivo_a;
    const char *archivo_b;
    /* Si esta en 1, se imprime C (matriz_imprimir) despues de cada
     * repeticion. Solo tiene sentido con matrices chicas. */
    int imprimir;
} ConfiguracionBenchmark;

int cli_parsear_argumentos(int argc, char **argv, ConfiguracionBenchmark *config,
                            const char *info_tamanio);

/* Llena A (n x m) y B (m x p) desde los archivos de config o con
 * matriz_llenar_aleatoria (semilla y semilla + 1). Devuelve 0 si todo
 * salio bien; si falla la lectura imprime el error en stderr y devuelve 1. */
int cli_cargar_matrices(const ConfiguracionBenchmark *config, real_t *a,
                        real_t *b, int n, int m, int p);

/* Encabezado de la corrida (CSV o texto). procesos <= 0 significa programa
 * serial: se omite la columna/campo procesos. detalle puede ser NULL; si no,
 * se agrega entre parentesis al nombre del algoritmo en el modo texto. */
void cli_imprimir_encabezado(const ConfiguracionBenchmark *config,
                             const char *algoritmo, const char *detalle,
                             int procesos, int n, int m, int p);

/* Linea de resultado de una repeticion; calcula los GFLOPS de 2*n*m*p. */
void cli_imprimir_repeticion(const ConfiguracionBenchmark *config,
                             const char *algoritmo, int procesos, int n, int m,
                             int p, int repeticion, double segundos,
                             double segundos_generacion);

#endif /* HPC_CLI_H */

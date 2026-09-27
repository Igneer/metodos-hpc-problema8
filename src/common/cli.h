#ifndef HPC_CLI_H
#define HPC_CLI_H

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

#endif /* HPC_CLI_H */

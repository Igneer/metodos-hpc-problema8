#ifndef HPC_CLI_H
#define HPC_CLI_H

typedef struct {
    unsigned semilla;
    int repeticiones;
    int csv;
} ConfiguracionBenchmark;

int cli_parsear_argumentos(int argc, char **argv, ConfiguracionBenchmark *config,
                            const char *info_tamanio);

#endif /* HPC_CLI_H */

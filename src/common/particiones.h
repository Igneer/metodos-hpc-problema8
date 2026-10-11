#ifndef HPC_PARTICIONES_H
#define HPC_PARTICIONES_H

/* size_t se utiliza para representar dimensiones y cantidades de elementos. */
#include <stddef.h>

typedef enum {
    /* La particion se calculo correctamente. */
    PARTICIONES_EXITO = 0,
    /* Hay un argumento invalido: procesos, capacidad o un puntero nulo. */
    PARTICIONES_ARGUMENTOS_INVALIDOS = 1,
    /* El arreglo fijo no tiene una posicion para cada proceso. */
    PARTICIONES_CAPACIDAD_INSUFICIENTE = 2,
    /* Un count o desplazamiento no puede representarse de forma segura. */
    PARTICIONES_DESBORDAMIENTO = 3
} EstadoParticiones;


int calcular_particiones(size_t filas, size_t columnas, int procesos,
                         int capacidad_procesos, int sendcounts[],
                         int displs[]);

#endif /* HPC_PARTICIONES_H */

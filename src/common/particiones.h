#ifndef HPC_PARTICIONES_H
#define HPC_PARTICIONES_H

/* size_t se utiliza para representar dimensiones y cantidades de elementos. */
#include <stddef.h>

/*
 * Codigos que describen el resultado de calcular_particiones().
 *
 * Ejemplo: si se piden 3 procesos y los arreglos solo tienen capacidad para
 * 2, la funcion devuelve PARTICIONES_CAPACIDAD_INSUFICIENTE en vez de escribir
 * fuera de los limites de los arreglos.
 */
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

/*
 * Calcula una particion balanceada de filas para una matriz row-major.
 *
 * `sendcounts` y `displs` quedan expresados en elementos, no en filas,
 * porque esa es la unidad que usan MPI_Scatterv y MPI_Gatherv.
 * Por ejemplo, 4 filas de una matriz con 10 columnas ocupan 40 elementos.
 *
 * Las filas sobrantes se asignan a los primeros procesos. Para 10 filas y 3
 * procesos, el reparto es 4, 3 y 3 filas.
 *
 * Los arreglos son proporcionados por el llamador y deben tener espacio para
 * `capacidad_procesos` posiciones. La funcion no reserva memoria dinamica.
 * En una futura llamada MPI, `columnas` sera M para A y P para C.
 */
int calcular_particiones(size_t filas, size_t columnas, int procesos,
                         int capacidad_procesos, int sendcounts[],
                         int displs[]);

#endif /* HPC_PARTICIONES_H */

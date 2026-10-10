#include "particiones.h"

#include <limits.h>
#include <stdint.h>

/*
   MPI usa int en la interfaz clasica de Scatterv/Gatherv. Los calculos de
   tamanio se hacen primero con size_t y solo se convierten despues de comprobar
   que el valor entra en el rango de int.
 */
static int convertir_a_int(size_t valor, int *resultado) {
    /* Ejemplo: un valor mayor que INT_MAX no puede guardarse en un count MPI. */
    if (valor > (size_t)INT_MAX) {
        return PARTICIONES_DESBORDAMIENTO;
    }

    /* El puntero apunta a sendcounts[rank] o displs[rank]. */
    *resultado = (int)valor;
    return PARTICIONES_EXITO;
}

/*
   Calcula la metadata de una particion 1D por filas.
   La funcion no mueve datos ni conoce MPI: solo llena sendcounts y displs.
   Esa separacion permite probar la aritmetica sin ejecutar procesos MPI.
 */
int calcular_particiones(size_t filas, size_t columnas, int procesos,
                         int capacidad_procesos, int sendcounts[],
                         int displs[]) {
    /* Antes de usar los arreglos comprobamos que la llamada sea valida. */
    if (procesos <= 0 || capacidad_procesos <= 0 || sendcounts == NULL ||
        displs == NULL) {
        return PARTICIONES_ARGUMENTOS_INVALIDOS;
    }

    /* Evita escribir sendcounts[procesos] fuera del arreglo fijo. */
    if (procesos > capacidad_procesos) {
        return PARTICIONES_CAPACIDAD_INSUFICIENTE;
    }

    /*
       Ejemplo con filas=10 y procesos=3:
       filas_base=3 y filas_sobrantes=1.
     */
    size_t cantidad_procesos = (size_t)procesos;
    size_t filas_base = filas / cantidad_procesos;
    size_t filas_sobrantes = filas % cantidad_procesos;

    /* El primer bloque empieza en el elemento cero. */
    size_t desplazamiento_actual = 0;

    /* Cada iteracion calcula el bloque que corresponde a un rank. */
    for (size_t rank = 0; rank < cantidad_procesos; rank++) {
        /* Los primeros filas_sobrantes ranks reciben una fila adicional. */
        size_t filas_locales = filas_base + (rank < filas_sobrantes ? 1 : 0);

        /*
           Antes de calcular filas_locales * columnas verificamos que la
           multiplicacion no desborde size_t.
         */
        if (columnas != 0 && filas_locales > SIZE_MAX / columnas) {
            return PARTICIONES_DESBORDAMIENTO;
        }

        /* MPI necesita cantidad de elementos, no cantidad de filas. */
        size_t elementos_locales = filas_locales * columnas;

        /*
           Para 10 columnas y el reparto 4,3,3 se obtienen counts 40,30,30.
           El segundo valor convertido es la posicion inicial del bloque.
         */
        if (convertir_a_int(elementos_locales, &sendcounts[rank]) !=
                PARTICIONES_EXITO ||
            convertir_a_int(desplazamiento_actual, &displs[rank]) !=
                PARTICIONES_EXITO) {
            return PARTICIONES_DESBORDAMIENTO;
        }

        /* Protege la suma que calcula el siguiente desplazamiento. */
        if (desplazamiento_actual > SIZE_MAX - elementos_locales) {
            return PARTICIONES_DESBORDAMIENTO;
        }

        /* El siguiente bloque comienza despues del bloque actual. */
        desplazamiento_actual += elementos_locales;
    }

    /* Todos los procesos tienen counts y displacements validos. */
    return PARTICIONES_EXITO;
}

#include "../src/common/particiones.h"

#include <assert.h>
#include <limits.h>
#include <stddef.h>

/*
 * Compara la salida del modulo con valores calculados manualmente.
 * Los arreglos esperados son const porque el test solo debe leerlos.
 */
static void comprobar_arreglos(const int *esperado_counts,
                               const int *esperado_displs, int procesos,
                               size_t filas, size_t columnas) {
    /*
     * Los buffers son estaticos: tienen ocho posiciones y no se usa malloc.
     * En una ejecucion MPI, la capacidad debe ser al menos el numero de ranks.
     */
    enum { CAPACIDAD_PROCESOS = 8 };
    int sendcounts[CAPACIDAD_PROCESOS];
    int displs[CAPACIDAD_PROCESOS];

    /* La llamada debe terminar correctamente para todos los casos validos. */
    assert(calcular_particiones(filas, columnas, procesos,
                                CAPACIDAD_PROCESOS, sendcounts, displs) ==
           PARTICIONES_EXITO);

    /*
     * Se verifica cada proceso y tambien que la suma cubra toda la matriz.
     * Para 10x10, la suma esperada es 40+30+30=100 elementos.
     */
    size_t total = 0;
    for (int rank = 0; rank < procesos; rank++) {
        assert(sendcounts[rank] == esperado_counts[rank]);
        assert(displs[rank] == esperado_displs[rank]);
        total += (size_t)sendcounts[rank];
    }
    assert(total == filas * columnas);
}

/* 10 filas, 3 procesos: 4,3,3 filas y counts 40,30,30. */
static void probar_reparto_no_divisible(void) {
    const int counts[] = {40, 30, 30};
    const int displs[] = {0, 40, 70};
    comprobar_arreglos(counts, displs, 3, 10, 10);
}

/* 12 filas, 4 procesos: todos reciben 3 filas de 5 elementos. */
static void probar_reparto_divisible(void) {
    const int counts[] = {15, 15, 15, 15};
    const int displs[] = {0, 15, 30, 45};
    comprobar_arreglos(counts, displs, 4, 12, 5);
}

/* Si hay mas procesos que filas, algunos reciben cero elementos. */
static void probar_mas_procesos_que_filas(void) {
    const int counts[] = {4, 4, 4, 0, 0};
    const int displs[] = {0, 4, 8, 12, 12};
    comprobar_arreglos(counts, displs, 5, 3, 4);
}

/* Los argumentos invalidos deben rechazarse antes de escribir los arreglos. */
static void probar_argumentos_invalidos(void) {
    int sendcounts[3];
    int displs[3];

    assert(calcular_particiones(10, 10, 0, 3, sendcounts, displs) ==
           PARTICIONES_ARGUMENTOS_INVALIDOS);
    assert(calcular_particiones(10, 10, 3, 3, NULL, displs) ==
           PARTICIONES_ARGUMENTOS_INVALIDOS);
    assert(calcular_particiones(10, 10, 3, 3, sendcounts, NULL) ==
           PARTICIONES_ARGUMENTOS_INVALIDOS);
    assert(calcular_particiones(10, 10, 3, 2, sendcounts, displs) ==
           PARTICIONES_CAPACIDAD_INSUFICIENTE);
}

/* INT_MAX es el limite de count/displ de la interfaz MPI clasica. */
static void probar_desbordamiento_de_count(void) {
    int sendcounts[1];
    int displs[1];
    size_t filas_fuera_de_mpi = (size_t)INT_MAX + 1;

    assert(calcular_particiones(filas_fuera_de_mpi, 1, 1,
                                1, sendcounts, displs) ==
           PARTICIONES_DESBORDAMIENTO);
}

/* Ejecuta todos los casos; un assert fallido hace fallar el test. */
int main(void) {
    probar_reparto_no_divisible();
    probar_reparto_divisible();
    probar_mas_procesos_que_filas();
    probar_argumentos_invalidos();
    probar_desbordamiento_de_count();
    return 0;
}

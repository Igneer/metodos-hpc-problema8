#ifndef HPC_MPI_COMUN_H
#define HPC_MPI_COMUN_H

#include <mpi.h>
#include <stdio.h>

#include "config.h"
#include "particiones.h"

/* Calcula como se reparte A (N x M) y C (N x P) por filas entre los procesos.
 * C tiene P columnas, no M: si P != M la particion cambia. Si algo falla,
 * el rank 0 informa y todos abortan. */
static inline void mpi_preparar_particiones(int rank, int size,
                                            int sendcounts[], int displs[],
                                            int recvcounts_C[], int displs_C[]) {
    if (size > MAX_PROCESOS) {
        if (rank == 0) {
            fprintf(stderr, "Error: %d procesos pedidos, MAX_PROCESOS=%d\n",
                    size, MAX_PROCESOS);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int estado = calcular_particiones((size_t)N, (size_t)M, size, MAX_PROCESOS,
                                      sendcounts, displs);
    if (estado != PARTICIONES_EXITO) {
        if (rank == 0) fprintf(stderr, "Error al calcular las particiones de A.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    estado = calcular_particiones((size_t)N, (size_t)P, size, MAX_PROCESOS,
                                  recvcounts_C, displs_C);
    if (estado != PARTICIONES_EXITO) {
        if (rank == 0) fprintf(stderr, "Error al calcular las particiones de C.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

#endif /* HPC_MPI_COMUN_H */

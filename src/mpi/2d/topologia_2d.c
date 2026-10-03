#include "topologia_2d.h"

#include <stddef.h>

/* Libera un comunicador solo si es valido. */
static void liberar_comm(MPI_Comm *comm) {
    if (*comm != MPI_COMM_NULL) MPI_Comm_free(comm);
}

int topologia_2d_crear(MPI_Comm base, Topologia2D *t) {
    if (t == NULL) return 1;
    t->grilla = t->fila = t->columna = MPI_COMM_NULL;

    int procesos;
    MPI_Comm_size(base, &procesos);

    /* dims en 0 deja que MPI elija la factorizacion mas equilibrada. */
    int dims[2] = {0, 0};
    MPI_Dims_create(procesos, 2, dims);
    t->filas_grilla = dims[0];
    t->columnas_grilla = dims[1];

    int periodica[2] = {0, 0};
    MPI_Cart_create(base, 2, dims, periodica, 1, &t->grilla);
    if (t->grilla == MPI_COMM_NULL) return 2;

    int coords[2];
    int rank;
    MPI_Comm_rank(t->grilla, &rank);
    MPI_Cart_coords(t->grilla, rank, 2, coords);
    t->fila_actual = coords[0];
    t->columna_actual = coords[1];

    /* remain_dims indica que dimension se conserva en el subgrupo. */
    int conserva_columnas[2] = {0, 1};
    int conserva_filas[2] = {1, 0};
    MPI_Cart_sub(t->grilla, conserva_columnas, &t->fila);
    MPI_Cart_sub(t->grilla, conserva_filas, &t->columna);

    int origen[2] = {0, 0};
    MPI_Cart_rank(t->grilla, origen, &t->rank_raiz);
    return 0;
}

void topologia_2d_liberar(Topologia2D *t) {
    if (t == NULL) return;
    liberar_comm(&t->fila);
    liberar_comm(&t->columna);
    liberar_comm(&t->grilla);
}

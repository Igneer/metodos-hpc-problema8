#ifndef HPC_TOPOLOGIA_2D_H
#define HPC_TOPOLOGIA_2D_H

#include <mpi.h>

/* Etapa 2 (cimiento), parte 1: grilla cartesiana y sus subcomunicadores.
 * Los ranks y las coordenadas los resuelve MPI; no hay formulas manuales. */
typedef struct {
    MPI_Comm grilla;      /* comunicador cartesiano Pr x Pc */
    MPI_Comm fila;        /* procesos de mi fila: varia la columna (A viaja aqui) */
    MPI_Comm columna;     /* procesos de mi columna: varia la fila (B viaja aqui) */
    int filas_grilla, columnas_grilla;
    int fila_actual, columna_actual;  /* mis coordenadas (MPI_Cart_coords) */
    int rank_raiz;                    /* rank en `grilla` de la coordenada (0,0) */
} Topologia2D;

/* Crea la grilla con MPI_Dims_create + MPI_Cart_create y los grupos de
 * filas/columnas con MPI_Cart_sub. Es colectiva sobre `base`.
 * Devuelve 0 si tuvo exito. Ownership: la topologia es duena de sus tres
 * comunicadores y el llamador debe invocar topologia_2d_liberar antes de
 * MPI_Finalize. Tras un error la estructura queda con comunicadores nulos.
 * Ejemplo: 6 procesos -> grilla 3x2. */
int topologia_2d_crear(MPI_Comm base, Topologia2D *topologia);

/* Libera los comunicadores (MPI_Comm_free) y los deja en MPI_COMM_NULL.
 * Es segura de llamar mas de una vez. */
void topologia_2d_liberar(Topologia2D *topologia);

#endif /* HPC_TOPOLOGIA_2D_H */

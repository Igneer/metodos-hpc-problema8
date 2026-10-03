#ifndef HPC_SUMMA_2D_H
#define HPC_SUMMA_2D_H

#include "kernel_local.h"
#include "particion_2d.h"
#include "topologia_2d.h"

/* Etapa 3: coreografia SUMMA sobre la infraestructura de la etapa 2.
 * Calcula C_local = sum_s A[:, segmento s] x B[segmento s, :] para cada
 * segmento de la dimension compartida, en el mismo orden en todos los procesos.
 *
 * Por segmento: el duenyo de ese tramo de A lo difunde por MPI_Bcast en su fila
 * (topologia->fila) y el duenyo del tramo de B en su columna
 * (topologia->columna); luego cada proceso invoca el kernel.
 * El duenyo transmite una vista con stride de su bloque local (un MPI_Type_vector
 * para A; B ya es contiguo) y los demas reciben un panel compacto: las firmas
 * (n_loc*ancho dobles) coinciden. Los duenyos operan sobre su bloque sin copias.
 *
 * A_local/B_local/C_local son los bloques contiguos de este proceso (etapa 2).
 * panel_a y panel_b son buffers del llamador con capacidad para el mayor panel
 * (N*M y M*P elementos bastan). C_local se pone en cero una vez al inicio.
 * Colectiva sobre la grilla. Devuelve 0 si tuvo exito. */
int summa_2d(const Topologia2D *topologia, const Particion2D *particion,
             KernelLocal kernel, const real_t *A_local, const real_t *B_local,
             real_t *C_local, real_t *panel_a, real_t *panel_b);

#endif /* HPC_SUMMA_2D_H */

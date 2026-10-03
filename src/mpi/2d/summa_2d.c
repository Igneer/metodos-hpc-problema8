#include "summa_2d.h"

#include <limits.h>

int summa_2d(const Topologia2D *t, const Particion2D *particion,
             KernelLocal kernel, const real_t *A_local, const real_t *B_local,
             real_t *C_local, real_t *panel_a, real_t *panel_b) {
    if (t == NULL || particion == NULL || kernel == NULL ||
        particion->num_segmentos == 0 ||
        particion->filas_a.partes != t->filas_grilla ||
        particion->columnas_b.partes != t->columnas_grilla) {
        return 1;
    }
    BloquesProceso2D mios;
    if (obtener_bloques_2d(particion, t->fila_actual, t->columna_actual, &mios)) return 1;

    size_t n_loc = mios.c.filas;          /* filas de A y de C */
    size_t p_loc = mios.c.columnas;       /* columnas de B y de C */
    size_t m_loc_a = mios.a.columnas;     /* ancho de mi bloque de A */

    /* C se inicializa una sola vez; cada segmento acumula encima. */
    for (size_t i = 0; i < n_loc * p_loc; i++) C_local[i] = 0.0;

    for (int s = 0; s < particion->num_segmentos; s++) {
        const Segmento2D *seg = &particion->segmentos[s];
        size_t ancho = seg->ancho;
        if (ancho > INT_MAX || n_loc * ancho > INT_MAX || ancho * p_loc > INT_MAX) return 1;

        /* A viaja por la fila: el duenyo es la columna propietaria. */
        const real_t *a_operando;
        size_t stride_a;
        MPI_Datatype tipo_a = MPI_DOUBLE;
        int derivado = 0;
        int count_a = (int)(n_loc * ancho);
        void *buf_a = panel_a;
        if (t->columna_actual == seg->propietario_columna_a) {
            a_operando = A_local + seg->offset_a;
            stride_a = m_loc_a;
            buf_a = (void *)a_operando;
            if (count_a > 0) {
                /* Vista con stride: n_loc filas de `ancho` dobles separadas
                 * m_loc_a; misma firma que n_loc*ancho dobles contiguos. */
                MPI_Type_vector((int)n_loc, (int)ancho, (int)m_loc_a, MPI_DOUBLE, &tipo_a);
                MPI_Type_commit(&tipo_a);
                derivado = 1;
                count_a = 1;
            }
        } else {
            a_operando = panel_a;
            stride_a = ancho;
        }
        MPI_Bcast(buf_a, count_a, tipo_a, seg->propietario_columna_a, t->fila);
        if (derivado) MPI_Type_free(&tipo_a);

        /* B viaja por la columna: filas enteras, ya contiguas en el duenyo. */
        const real_t *b_operando;
        void *buf_b;
        if (t->fila_actual == seg->propietario_fila_b) {
            b_operando = B_local + seg->offset_b * p_loc;
            buf_b = (void *)b_operando;
        } else {
            b_operando = panel_b;
            buf_b = panel_b;
        }
        MPI_Bcast(buf_b, (int)(ancho * p_loc), MPI_DOUBLE, seg->propietario_fila_b,
                  t->columna);

        kernel(a_operando, b_operando, C_local, n_loc, ancho, p_loc,
               stride_a, p_loc, p_loc);
    }
    return 0;
}

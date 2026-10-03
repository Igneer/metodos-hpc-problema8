#include "transferencias_2d.h"

#include <limits.h>

enum { TAG_DISTRIBUIR = 21, TAG_RECONSTRUIR = 22 };

Bloque2D bloque_de_matriz_2d(const BloquesProceso2D *bloques, Matriz2D cual) {
    switch (cual) {
        case MATRIZ_2D_A: return bloques->a;
        case MATRIZ_2D_B: return bloques->b;
        default:          return bloques->c;
    }
}

int tipo_bloque_2d_crear(const Bloque2D *bloque, size_t filas_globales,
                         size_t columnas_globales, TipoBloque2D *tipo) {
    if (bloque == NULL || tipo == NULL) return 1;
    tipo->tipo = MPI_DOUBLE;
    tipo->count = 0;
    if (filas_globales > INT_MAX || columnas_globales > INT_MAX) return 1;
    if (bloque->filas == 0 || bloque->columnas == 0) return 0;  /* vacio */

    int tamanios[2] = {(int)filas_globales, (int)columnas_globales};
    int subtamanios[2] = {(int)bloque->filas, (int)bloque->columnas};
    int inicios[2] = {(int)bloque->inicio_fila, (int)bloque->inicio_columna};
    MPI_Type_create_subarray(2, tamanios, subtamanios, inicios, MPI_ORDER_C,
                             MPI_DOUBLE, &tipo->tipo);
    MPI_Type_commit(&tipo->tipo);
    tipo->count = 1;
    return 0;
}

void tipo_bloque_2d_liberar(TipoBloque2D *tipo) {
    if (tipo != NULL && tipo->count > 0) MPI_Type_free(&tipo->tipo);
    if (tipo != NULL) {
        tipo->tipo = MPI_DOUBLE;
        tipo->count = 0;
    }
}

/* Dimensiones globales de la matriz segun cual. */
static void dimensiones_globales(const Particion2D *p, Matriz2D cual,
                                 size_t *filas, size_t *columnas) {
    switch (cual) {
        case MATRIZ_2D_A: *filas = p->n; *columnas = p->m; break;
        case MATRIZ_2D_B: *filas = p->m; *columnas = p->p; break;
        default:          *filas = p->n; *columnas = p->p; break;
    }
}

/* Cuerpo comun de distribuir/reconstruir: la raiz recorre todos los procesos.
 * `hacia_locales` = 1 distribuye (global -> local), 0 reconstruye. */
static int transferir(const Topologia2D *t, const Particion2D *particion,
                      Matriz2D cual, int hacia_locales,
                      const real_t *global_lectura, real_t *global_escritura,
                      const real_t *local_lectura, real_t *local_escritura) {
    if (t == NULL || particion == NULL || particion->num_segmentos == 0 ||
        particion->filas_a.partes != t->filas_grilla ||
        particion->columnas_b.partes != t->columnas_grilla) {
        return 1;
    }
    size_t filas_g, columnas_g;
    dimensiones_globales(particion, cual, &filas_g, &columnas_g);

    int rank;
    MPI_Comm_rank(t->grilla, &rank);
    int tag = hacia_locales ? TAG_DISTRIBUIR : TAG_RECONSTRUIR;

    /* Mi bloque: cuantos elementos contiguos tiene mi buffer local. */
    BloquesProceso2D mios;
    if (obtener_bloques_2d(particion, t->fila_actual, t->columna_actual, &mios)) return 1;
    Bloque2D mi_bloque = bloque_de_matriz_2d(&mios, cual);
    size_t elementos_locales = mi_bloque.filas * mi_bloque.columnas;
    if (elementos_locales > INT_MAX) return 1;

    if (rank != t->rank_raiz) {
        if (hacia_locales) {
            MPI_Recv(local_escritura, (int)elementos_locales, MPI_DOUBLE,
                     t->rank_raiz, tag, t->grilla, MPI_STATUS_IGNORE);
        } else {
            MPI_Send(local_lectura, (int)elementos_locales, MPI_DOUBLE,
                     t->rank_raiz, tag, t->grilla);
        }
        return 0;
    }

    int estado = 0;
    int total = t->filas_grilla * t->columnas_grilla;
    for (int r = 0; r < total; r++) {
        int coords[2];
        MPI_Cart_coords(t->grilla, r, 2, coords);
        BloquesProceso2D bloques;
        if (obtener_bloques_2d(particion, coords[0], coords[1], &bloques)) {
            estado = 1;
            continue;
        }
        Bloque2D b = bloque_de_matriz_2d(&bloques, cual);
        TipoBloque2D tipo;
        if (tipo_bloque_2d_crear(&b, filas_g, columnas_g, &tipo)) {
            estado = 1;
            continue;
        }
        int elementos = (int)(b.filas * b.columnas);

        if (r == rank) {
            /* Copia propia: el mismo mecanismo MPI, sin enviarse a si misma. */
            if (hacia_locales) {
                MPI_Sendrecv(global_lectura, tipo.count, tipo.tipo, r, tag,
                             local_escritura, elementos, MPI_DOUBLE, r, tag,
                             t->grilla, MPI_STATUS_IGNORE);
            } else {
                MPI_Sendrecv(local_lectura, elementos, MPI_DOUBLE, r, tag,
                             global_escritura, tipo.count, tipo.tipo, r, tag,
                             t->grilla, MPI_STATUS_IGNORE);
            }
        } else if (hacia_locales) {
            MPI_Send(global_lectura, tipo.count, tipo.tipo, r, tag, t->grilla);
        } else {
            MPI_Recv(global_escritura, tipo.count, tipo.tipo, r, tag, t->grilla,
                     MPI_STATUS_IGNORE);
        }
        tipo_bloque_2d_liberar(&tipo);
    }
    return estado;
}

int distribuir_2d(const Topologia2D *topologia, const Particion2D *particion,
                  Matriz2D cual, const real_t *global, real_t *local) {
    return transferir(topologia, particion, cual, 1, global, NULL, NULL, local);
}

int reconstruir_2d(const Topologia2D *topologia, const Particion2D *particion,
                   Matriz2D cual, const real_t *local, real_t *global) {
    return transferir(topologia, particion, cual, 0, NULL, global, local, NULL);
}

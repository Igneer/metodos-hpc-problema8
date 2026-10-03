#ifndef HPC_TRANSFERENCIAS_2D_H
#define HPC_TRANSFERENCIAS_2D_H

#include <mpi.h>

#include "../../common/matrix.h"
#include "particion_2d.h"
#include "topologia_2d.h"

/* Etapa 2 (cimiento), parte 2: datatypes de bloque, distribucion inicial y
 * reconstruccion. Usa los descriptores de la etapa 1; no multiplica. */

typedef enum { MATRIZ_2D_A, MATRIZ_2D_B, MATRIZ_2D_C } Matriz2D;

/* Un bloque de una matriz global visto como mensaje MPI.
 * Bloque no vacio: `tipo` es un subarray confirmado y count=1.
 * Bloque vacio: tipo=MPI_DOUBLE (basico, valido) y count=0.
 * Ownership: quien lo crea llama tipo_bloque_2d_liberar. */
typedef struct {
    MPI_Datatype tipo;
    int count;
} TipoBloque2D;

/* Describe `bloque` dentro de una matriz global row-major de
 * filas_globales x columnas_globales (MPI_Type_create_subarray + commit).
 * Ejemplo: A01 de la 4x4 es subarray sizes {4,4}, subsizes {2,2}, starts {0,2};
 * al enviarlo salen los valores 3, 4, 7 y 8. Devuelve 0 si tuvo exito. */
int tipo_bloque_2d_crear(const Bloque2D *bloque, size_t filas_globales,
                         size_t columnas_globales, TipoBloque2D *tipo);

/* Libera el datatype derivado (MPI_Type_free); no toca los basicos. */
void tipo_bloque_2d_liberar(TipoBloque2D *tipo);

/* Distribucion inicial (colectiva sobre topologia->grilla): la raiz envia a
 * cada proceso su bloque de la matriz `cual`; `local` recibe el bloque
 * contiguo (stride = columnas del bloque). `global` solo se lee en la raiz y
 * puede ser NULL en los demas. El llamador garantiza capacidad en `local`.
 * Devuelve 0 si tuvo exito. */
int distribuir_2d(const Topologia2D *topologia, const Particion2D *particion,
                  Matriz2D cual, const real_t *global, real_t *local);

/* Operacion inversa: la raiz reconstruye `global` desde los bloques locales.
 * `global` solo se escribe en la raiz. Devuelve 0 si tuvo exito. */
int reconstruir_2d(const Topologia2D *topologia, const Particion2D *particion,
                   Matriz2D cual, const real_t *local, real_t *global);

/* Selecciona el bloque de la matriz indicada dentro de un BloquesProceso2D. */
Bloque2D bloque_de_matriz_2d(const BloquesProceso2D *bloques, Matriz2D cual);

#endif /* HPC_TRANSFERENCIAS_2D_H */

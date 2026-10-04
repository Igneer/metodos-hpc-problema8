#ifndef HPC_BLOQUE_H
#define HPC_BLOQUE_H

/* Tamanio de bloque para cache blocking/tiling de ikj.
 *
 * Por defecto BI = BK = BJ = 64 (bloque de B de 64x64 doubles = 32 KiB).
 * Para otro tamanio, compilar con -DBI=.. -DBK=.. -DBJ=.. (o -DBLOQUE=..
 * para los tres).
 *
 * No se deriva de la cache: el optimo depende de la jerarquia de cada
 * maquina y una formula basada en la L2 no funciono en las dos
 * probadas (scripts/barrido_bloque.sh, GFLOPS del serial bloqueado):
 *   - AMD EPYC 7B12 (L1d 32 KiB, L2 512 KiB): meseta en b = 144..192,
 *     ~10-18% sobre b = 64.
 *   - Intel Core Ultra 9 185H (L1d 48 KiB, L2 2 MiB): optimo en b = 64;
 *     b >= 192 pierde 25-30%.
 * 64 es el valor con el que se midieron los benchmarks y es el optimo en
 * una de las maquinas. Ver informe, seccion 2.3.
 */

#include "matrix.h"

#ifndef BLOQUE
#define BLOQUE 64
#endif

#ifndef BI
#define BI BLOQUE
#endif
#ifndef BK
#define BK BLOQUE
#endif
#ifndef BJ
#define BJ BLOQUE
#endif

typedef struct {
    int bi, bk, bj;
} TamanioBloque;

static TamanioBloque bloque_calcular(void) {
    TamanioBloque t = {BI, BK, BJ};
    return t;
}

#endif /* HPC_BLOQUE_H */

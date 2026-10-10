#ifndef HPC_BLOQUE_H
#define HPC_BLOQUE_H

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

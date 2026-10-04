#ifndef HPC_BLOQUE_H
#define HPC_BLOQUE_H

/* Tamanio de bloque para cache blocking/tiling de ikj.
 *
 * Si se compila con -DBI/-DBK/-DBJ se usan esos valores fijos. Si no,
 * el tamanio se calcula en runtime a partir de la cache L2 de la
 * maquina (sysconf, o L2_BYTES_DEFAULT si no se puede consultar): se
 * usa como mucho la mitad de L2 para los tres bloques activos
 * (BI*BK de A, BK*BJ de B, BI*BJ de C), dejando el resto para el
 * resto de los datos y otros procesos. Con bloques cuadrados b x b:
 *   3 * b * b * sizeof(real_t) <= L2 / 2
 * y se redondea hacia abajo a multiplo de 8 doubles (una linea de
 * cache de 64 bytes), con minimo 16 y maximo 512.
 */

#include <unistd.h>

#include "matrix.h"

#ifndef L2_BYTES_DEFAULT
#define L2_BYTES_DEFAULT (256 * 1024)
#endif

typedef struct {
    int bi, bk, bj;
} TamanioBloque;

static long bloque_bytes_l2(void) {
    long l2 = 0;
#ifdef _SC_LEVEL2_CACHE_SIZE
    l2 = sysconf(_SC_LEVEL2_CACHE_SIZE);
#endif
    return l2 > 0 ? l2 : L2_BYTES_DEFAULT;
}

static TamanioBloque bloque_calcular(void) {
    int b = 0;
    long l2 = bloque_bytes_l2();
    long maximo = l2 / 2 / (3 * (long)sizeof(real_t)); /* b*b <= maximo */
    while ((long)(b + 8) * (b + 8) <= maximo && b + 8 <= 512) b += 8;
    if (b < 16) b = 16;

    TamanioBloque t = {b, b, b};
#ifdef BI
    t.bi = BI;
#endif
#ifdef BK
    t.bk = BK;
#endif
#ifdef BJ
    t.bj = BJ;
#endif
    return t;
}

#endif /* HPC_BLOQUE_H */

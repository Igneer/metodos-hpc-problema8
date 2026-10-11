#ifndef HPC_KERNELS_H
#define HPC_KERNELS_H

#include "bloque.h"
#include "matrix.h"

static inline int min_int(int a, int b) {
    return a < b ? a : b;
}

static inline void multiplicar_ijk(const real_t *restrict a, const real_t *restrict b,
                                   real_t *restrict c, int n, int m, int p) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            real_t acumulador = 0.0;
            for (int k = 0; k < m; k++) {
                acumulador += a[(size_t)i * m + k] * b[(size_t)k * p + j];
            }
            c[(size_t)i * p + j] = acumulador;
        }
    }
}

static inline void multiplicar_ikj(const real_t *restrict a, const real_t *restrict b,
                                   real_t *restrict c, int n, int m, int p) {
    for (int i = 0; i < n; i++) {
        real_t *restrict fila_c = c + (size_t)i * p;
        for (int k = 0; k < m; k++) {
            const real_t a_ik = a[(size_t)i * m + k];
            const real_t *restrict fila_b = b + (size_t)k * p;
            for (int j = 0; j < p; j++) {
                fila_c[j] += a_ik * fila_b[j];
            }
        }
    }
}

/* Version bloqueada de ikj: los tres loops externos (ii, jj, kk)
   recorren en bloques BI x BK x BJ; el cuerpo interno es el mismo ikj
   de siempre, acotado al bloque actual. c debe venir en cero. */
static inline void multiplicar_ikj_bloqueada(const real_t *restrict a,
                                             const real_t *restrict b,
                                             real_t *restrict c,
                                             int n, int m, int p,
                                             TamanioBloque tb) {
    const int bi = tb.bi, bk = tb.bk, bj = tb.bj;
    for (int ii = 0; ii < n; ii += bi) {
        int i_fin = min_int(ii + bi, n);
        for (int jj = 0; jj < p; jj += bj) {
            int j_fin = min_int(jj + bj, p);
            for (int kk = 0; kk < m; kk += bk) {
                int k_fin = min_int(kk + bk, m);

                for (int i = ii; i < i_fin; i++) {
                    real_t *restrict fila_c = c + (size_t)i * p;
                    for (int k = kk; k < k_fin; k++) {
                        const real_t a_ik = a[(size_t)i * m + k];
                        const real_t *restrict fila_b = b + (size_t)k * p;
                        for (int j = jj; j < j_fin; j++) {
                            fila_c[j] += a_ik * fila_b[j];
                        }
                    }
                }
            }
        }
    }
}

#endif /* HPC_KERNELS_H */

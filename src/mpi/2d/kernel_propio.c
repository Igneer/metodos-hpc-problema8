#include "kernel_propio.h"

void kernel_propio_ikj(const real_t *restrict a, const real_t *restrict b,
                       real_t *restrict c, size_t filas, size_t ancho,
                       size_t columnas, size_t stride_a, size_t stride_b,
                       size_t stride_c) {
    if (filas == 0 || ancho == 0 || columnas == 0) return;
    for (size_t i = 0; i < filas; i++) {
        real_t *restrict fila_c = c + i * stride_c;
        for (size_t k = 0; k < ancho; k++) {
            const real_t a_ik = a[i * stride_a + k];
            const real_t *restrict fila_b = b + k * stride_b;
            for (size_t j = 0; j < columnas; j++) {
                fila_c[j] += a_ik * fila_b[j];
            }
        }
    }
}

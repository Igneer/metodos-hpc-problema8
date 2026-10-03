#ifndef HPC_KERNEL_PROPIO_H
#define HPC_KERNEL_PROPIO_H

#include "kernel_local.h"

/* Backend propio (sin OpenBLAS) del contrato KernelLocal: C += A*B con
 * orden ikj y strides explicitos. Sin MPI ni reservas de memoria.
 * Si alguna dimension es cero no accede a los buffers. */
void kernel_propio_ikj(const real_t *restrict a, const real_t *restrict b,
                       real_t *restrict c, size_t filas, size_t ancho,
                       size_t columnas, size_t stride_a, size_t stride_b,
                       size_t stride_c);

#endif /* HPC_KERNEL_PROPIO_H */

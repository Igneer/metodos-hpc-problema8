#ifndef HPC_KERNEL_LOCAL_H
#define HPC_KERNEL_LOCAL_H

#include "../../common/matrix.h"

/* "KernelLocal" nombra la operacion numerica dentro de un solo proceso.
 * En la etapa 1 se define este tipo de funcion; en la etapa 2 se implementara
 * un backend; en la etapa 3 SUMMA lo invocara tras disponer de ambos paneles.
 * Es un puntero a funcion, no una funcion implementada ni un algoritmo MPI.
 * Contrato para futuros backends: C += A*B, sin MPI ni reserva de matrices.
 * A: filas x ancho; B: ancho x columnas; C: filas x columnas, row-major.
 * Los strides son distancias entre comienzos de filas, en elementos:
 * stride_a >= ancho; stride_b y stride_c >= columnas.
 * El llamador garantiza buffers suficientes y C no solapado con A/B.
 * C conserva su valor anterior; se pone en cero una vez fuera del kernel.
 * Si alguna dimension es cero, la operacion no accede a los buffers.
 * Ejemplo: un panel de dos columnas dentro de una fila de cuatro usa
 * stride_a=4; el mismo panel recibido compacto usa stride_a=2.
 * OpenBLAS traducira este contrato a GEMM con alpha=1 y beta=1.
 * Este header solo define la interfaz; no implementa ningun backend. */
typedef void (*KernelLocal)(const real_t *restrict a,
                            const real_t *restrict b,
                            real_t *restrict c,
                            size_t filas, size_t ancho, size_t columnas,
                            size_t stride_a, size_t stride_b, size_t stride_c);

#endif /* HPC_KERNEL_LOCAL_H */

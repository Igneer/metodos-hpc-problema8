#ifndef HPC_TIMING_H
#define HPC_TIMING_H

/* clock_gettime/CLOCK_MONOTONIC son POSIX, no ISO C: con -std=c11
 * estricto glibc los esconde salvo que se pida soporte POSIX antes
 * del primer header del sistema incluido en el .c (por eso este
 * header se incluye primero en los .c que lo usan). */
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 199309L
#endif
#include <time.h>

static inline double ahora_segundos(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

#endif /* HPC_TIMING_H */

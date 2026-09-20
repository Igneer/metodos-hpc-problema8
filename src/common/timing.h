#ifndef HPC_TIMING_H
#define HPC_TIMING_H

#include <time.h>

static inline double ahora_segundos(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

#endif /* HPC_TIMING_H */

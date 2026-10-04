#ifndef HPC_CONFIG_H
#define HPC_CONFIG_H


#ifndef N
#define N 512
#endif

#ifndef M
#define M 512
#endif

#ifndef P
#define P 512
#endif

/* Cantidad maxima de procesos MPI que soportan los arreglos estaticos
 * de sendcounts/displs en src/mpi/*.c */
#ifndef MAX_PROCESOS
#define MAX_PROCESOS 1024
#endif

#endif /* HPC_CONFIG_H */

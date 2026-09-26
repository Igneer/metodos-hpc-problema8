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

/* Tamanio de bloque para la version con cache blocking/tiling de
 * ikj (ver src/serial/matmul_ikj_bloqueado.c). Los tres bloques
 * activos a la vez (BI*BK de A, BK*BJ de B, BI*BJ de C) deben entrar
 * juntos en la cache objetivo (tipicamente L2): 3 bloques de 64x64
 * doubles ~= 98KB, razonable para L2 de 256KB+. Ajustar segun la
 * cache real de la maquina de prueba. */
#ifndef BI
#define BI 64
#endif

#ifndef BK
#define BK 64
#endif

#ifndef BJ
#define BJ 64
#endif

/* Cantidad maxima de procesos MPI que soportan los arreglos estaticos
 * de sendcounts/displs en src/mpi/*.c */
#ifndef MAX_PROCESOS
#define MAX_PROCESOS 1024
#endif

#endif /* HPC_CONFIG_H */

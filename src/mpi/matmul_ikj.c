#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "../common/particiones.h"
#include "../common/matrix.h"

static void multiplicar_ikj(const real_t *restrict a, const real_t *restrict b,
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

int main(int argc, char *argv[]) {
    int rank, size;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    MPI_Comm_size(MPI_COMM_WORLD, &size); 

    /* 
     * Definimos las dimensiones del problema. 
     * (M x K) multiplicada por (K x N) da como resultado (M x N)
     */
    size_t M = 2000; 
    size_t K = 2000; 
    size_t N = 2000; 

    if (rank == 0) {
        printf("Iniciando cálculo MPI con %d procesos. Matrices de %zux%zu...\n", size, M, N);
    }

    int *sendcounts = malloc(size * sizeof(int)); /*Elementos que le tocan a cada proceso */
    int *displs = malloc(size * sizeof(int));/*Posiciones iniciales de cada proceso */

    /* 
     * Particionamos la matriz entre los procesos 
     */
    int estado = calcular_particiones(M, K, size, size, sendcounts, displs);

    if (estado != PARTICIONES_EXITO) {
        if (rank == 0) {
            printf("Error fatal al calcular las particiones.\n");
        }
        free(sendcounts);
        free(displs);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* 
     * Cada revisa cuantos elementos de A le tocaron
     * Luego se calulan las filas dividiendo en las columnas
     */
    int mis_elementos_A = sendcounts[rank];
    size_t mis_filas = mis_elementos_A / K; 

    /* printf("Proceso %d listo: recibo %d elementos de A (%zu filas)\n", rank, mis_elementos_A, mis_filas); */

    real_t *A = NULL;
    real_t *C = NULL;
    
    /* 
     * La matriz B se reserva completa en todos los procesos. 
     * Para hacer el producto necesito ver toda la matriz B
     */
    real_t *B = malloc(K * N * sizeof(real_t));

    if (rank == 0) {
        /* El proceso 0 (root) reserva A y C completas en la memoria */
        A = malloc(M * K * sizeof(real_t));
        C = malloc(M * N * sizeof(real_t));

        printf("Proceso 0: Inicializando datos...\n");
        
        matriz_llenar_aleatoria(A, M, K, 1234); 
        matriz_llenar_aleatoria(B, K, N, 5678); 
        
        matriz_poner_cero(C, M, N);             
    }

    /* 
     * Procedemos a calcular el tamanio de la variable local A y C que le corresponde a cada proceso.
     */
    real_t *local_A = malloc(mis_elementos_A * sizeof(real_t));
    
    int mis_elementos_C = mis_filas * N;
    real_t *local_C = malloc(mis_elementos_C * sizeof(real_t));
    
    if (mis_elementos_C > 0) {
        matriz_poner_cero(local_C, mis_filas, N);
    }

    if (rank == 0) {
        printf("Repartiendo los datos a los procesos...\n");
    }
    double tiempo_inicio = MPI_Wtime(); 
    /* 
     * Broadcast de la matriz B
     * Hacemos una copia de la matriz B en todos los procesos.
     * Como 'real_t' esta definido como 'double', usamos MPI_DOUBLE.
     */
    MPI_Bcast(B, K * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* 
     * Scatter de la matriz A
     * MPI_Scatterv toma la matriz completa A (solo válida en el root) y la reparte
     * usando los tamaniosos que calculamos calculando las particiones.
     */
    MPI_Scatterv(
        A,                  /* matriz completa que quiero particionar */
        sendcounts,         /* cuantos elementos voy a enviar a cada proceso */
        displs,             /* desde que posicion voy a enviar a cada proceso */
        MPI_DOUBLE,         /* Tipo de dato que enviamos*/
        local_A,            /* Buffer de recepcion (variable donde recibo mi "porcion") */
        mis_elementos_A,    /* Cuantos elementos recibe cada proceso */
        MPI_DOUBLE,         /* Tipo de dato que recibimos */
        0,                  /* Rank del proceso que envía (root) */
        MPI_COMM_WORLD      /* Comunicador */
    );
    /* 
     * Hacemos la multiplicacion usando el orden i-k-j.
     */

    multiplicar_ikj(local_A, B, local_C, mis_filas, K, N);
    if (rank == 0) {
        printf("Cálculo terminado. Recolectando resultados...\n");
    }

    /* 
     * La matriz C tiene N columnas, mientras que A tenia K.
     * Si N y K son diferentes, la cantidad de elementos a recolectar cambia.
     * Tenemos que volver a calcular los 'counts' y 'displacements' para la matriz C.
     */
    int *recvcounts_C = malloc(size * sizeof(int));
    int *displs_C = malloc(size * sizeof(int));
    
    estado = calcular_particiones(M, N, size, size, recvcounts_C, displs_C);
    if (estado != PARTICIONES_EXITO) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* 
     * Tomamos los resultados obtenidos en cada proceso guardado en local_C y los recolectamos
     * en la matriz C del proceso root.
     */
    MPI_Gatherv(
        local_C,            /* Buffer de envío (lo que calculo el proceso) */
        mis_elementos_C,    /* los elemento que calculo el proceso */
        MPI_DOUBLE,         /* Tipo de dato */
        C,                  /* Buffer de recepción (La matriz final del resultado) */
        recvcounts_C,       /* cuantos elementos manda cada proceso */
        displs_C,           /* posicion donde empieza los resultados de cada proceso */
        MPI_DOUBLE,         /* Tipo de dato */
        0,                  /* Rank del proceso que recolecta (root) */
        MPI_COMM_WORLD      /* Comunicador */
    );

    double tiempo_fin = MPI_Wtime();

    if (rank == 0) {
        double tiempo_total = tiempo_fin - tiempo_inicio;
        printf("Cálculo completado en %f segundos.\n", tiempo_total);
    }

    
    free(sendcounts);
    free(displs);
    free(recvcounts_C);
    free(displs_C);
    
    free(local_A);
    free(local_C);
    free(B);
    
    if (rank == 0) {
        free(A);
        free(C);
    }
    
    MPI_Finalize();
    return 0;
}
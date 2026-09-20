#ifndef HPC_MATRIX_H
#define HPC_MATRIX_H

#include <stddef.h>

/* Tipo base de las matrices. Centralizado aca para poder pasar a
 * `float` mas adelante (comparar precision simple vs doble, o
 * duplicar el ancho de SIMD) cambiando una sola linea. */
typedef double real_t;

/*
 * Estas funciones son agnosticas de como este reservada la memoria
 * (estatica, en este caso): reciben un puntero + dimensiones y
 * operan sobre un bloque contiguo row-major (fila 0 completa, luego
 * fila 1, ...). Ese layout es el que permite accesos secuenciales
 * dentro de una fila (ver multiplicar_ikj en matmul_ikj.c), y es el
 * mismo que va a necesitar MPI para repartir bloques de filas con
 * Scatterv/Gatherv sin conversiones extra.
 */

/* Llena matriz (filas*columnas elementos) con valores uniformes en
 * [0, 1), usando `semilla` para reproducibilidad. Cada matriz de una
 * corrida debe llamarse con una semilla distinta (ej. semilla y
 * semilla+1 para A y B) para que no queden identicas. */
void matriz_llenar_aleatoria(real_t *matriz, size_t filas, size_t columnas,
                              unsigned semilla);

/* Pone en cero los filas*columnas elementos de matriz (usado para
 * resetear el acumulador C entre repeticiones de benchmark). */
void matriz_poner_cero(real_t *matriz, size_t filas, size_t columnas);

/* Imprime la matriz por pantalla (stdout), una fila por linea, con
 * columnas alineadas. Pensada para matrices chicas (debug/tests);
 * para matrices grandes (512+) el volumen de texto la hace poco
 * practica. */
void matriz_imprimir(const real_t *matriz, size_t filas, size_t columnas);

/* Exporta la matriz a un archivo de texto plano: una fila por linea,
 * valores separados por espacios. Devuelve 0 si pudo escribir el
 * archivo completo, distinto de 0 si fallo (por ejemplo no se pudo
 * abrir `ruta` para escritura). */
int matriz_exportar_archivo(const real_t *matriz, size_t filas, size_t columnas,
                             const char *ruta);

/* Lee una matriz desde un archivo de texto plano generado con
 * matriz_exportar_archivo (o con el mismo formato: valores separados
 * por espacios/saltos de linea). `matriz` ya tiene que estar
 * reservada por el llamador con espacio para filas*columnas
 * elementos; esta funcion no conoce las dimensiones del archivo, solo
 * lee filas*columnas valores de el. Devuelve 0 si pudo leer todos los
 * valores esperados, distinto de 0 si fallo (archivo inexistente o
 * con menos valores de los esperados). */
int matriz_leer_archivo(real_t *matriz, size_t filas, size_t columnas,
                         const char *ruta);

#endif /* HPC_MATRIX_H */

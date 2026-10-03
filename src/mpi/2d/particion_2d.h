#ifndef HPC_PARTICION_2D_H
#define HPC_PARTICION_2D_H

#include "../../common/config.h"
#include "../../common/particiones.h"

/* Metadata fija de la etapa 1: no contiene matrices, comunicadores ni ranks.
 * "Eje" representa una dimension repartida, antes de combinarla en bloques. */
typedef struct {
    int partes;
    int tamanios[MAX_PROCESOS];
    int inicios[MAX_PROCESOS];
} EjeParticion2D;

/* Rectangulo de la matriz global: describe ubicacion y forma, no sus valores. */
typedef struct {
    size_t filas, columnas;
    size_t inicio_fila, inicio_columna;
} Bloque2D;

/* Los tres rectangulos asociados a la misma coordenada logica de grilla. */
typedef struct {
    Bloque2D a, b, c;
} BloquesProceso2D;

/* Un tramo [inicio, inicio+ancho) del eje compartido entre A y B.
 * Los propietarios son coordenadas, no identificadores de procesos. */
typedef struct {
    size_t inicio, ancho;
    int propietario_columna_a;
    int propietario_fila_b;
    size_t offset_a, offset_b;
} Segmento2D;

typedef struct {
    size_t n, m, p;
    EjeParticion2D filas_a;       /* N dividido entre filas de grilla. */
    EjeParticion2D columnas_a;    /* M dividido entre columnas de grilla. */
    EjeParticion2D filas_b;       /* M dividido entre filas de grilla. */
    EjeParticion2D columnas_b;    /* P dividido entre columnas de grilla. */
    int num_segmentos;
    /* Pr+Pc-1 <= Pr*Pc <= MAX_PROCESOS: esta capacidad cubre los cortes. */
    Segmento2D segmentos[MAX_PROCESOS];
} Particion2D;

/* Entrada principal: usa N, M y P definidos en config.h (o con -DN/-DM/-DP).
 * Solo recibe la grilla y la estructura de salida; el proyecto conserva sus
 * dimensiones de compilacion. Delega el calculo y los errores a la funcion
 * con dimensiones explicitas. No reserva ni transfiere matrices. */
int calcular_particion_2d(int filas_grilla, int columnas_grilla,
                         Particion2D *particion);

/* Variante con dimensiones explicitas, para reutilizacion y pruebas de casos
 * diferentes en una misma ejecucion. Etapa 1: calcular el plan de reparto.
 * Describe A(NxM), B(MxP), C(NxP); reutiliza calcular_particiones().
 * Las dimensiones globales deben ser positivas; los bloques pueden ser vacios.
 * La grilla completa debe caber en MAX_PROCESOS. Dimensiones y counts locales
 * se limitan a INT_MAX para la futura interfaz MPI clasica.
 * El llamador proporciona la estructura (preferentemente static). No se reserva
 * memoria ni se mueve informacion. Devuelve los codigos EstadoParticiones.
 * Tras un error, num_segmentos queda en cero y la particion no es utilizable.
 * Ejemplo: M=6, Pr=2, Pc=3 produce anchos de segmento 2,1,1,2. */
int calcular_particion_2d_dimensiones(size_t n, size_t m, size_t p,
                                     int filas_grilla, int columnas_grilla,
                                     Particion2D *particion);

/* Consulta el plan de la etapa 1: obtiene descriptores, no matrices.
 * Obtiene bloques por coordenadas logicas de una particion exitosa.
 * MPI_Cart_coords proporcionara esas coordenadas en la etapa de topologia.
 * Ejemplo: A01 de una matriz 4x4 tiene inicio (0,2) y tamanio 2x2.
 * El resultado solo se modifica si la consulta tiene exito. */
int obtener_bloques_2d(const Particion2D *particion,
                      int fila_grilla, int columna_grilla,
                      BloquesProceso2D *bloques);

#endif /* HPC_PARTICION_2D_H */

#include "particion_2d.h"

#include <limits.h>
#include <stdint.h>

/* Flujo de este modulo, perteneciente a la etapa 1 (estructura):
 * calcular_particion_2d -> N/M/P de config.h -> calcular_particion_2d_dimensiones
 *   -> preparar_eje -> calcular_particiones (1D existente)
 *   -> count_local_valido -> segmentos comunes de M.
 * obtener_bloques_2d   -> describir_bloque para A, B y C.
 * La etapa 2 usara estos descriptores para crear datatypes y mover matrices;
 * la etapa 3 recorrera los segmentos para ejecutar SUMMA. Aqui solo se
 * describen datos: no se crean procesos, no se comunica y no se multiplica. */

/* Helper interno de la etapa 1: prepara la particion de una sola dimension.
 * Se llama "preparar_eje" porque llena tamanios e inicios sobre un eje,
 * no construye una matriz ni una topologia. Reutiliza la particion 1D con
 * columnas=1 para expresar sus resultados en posiciones del eje.
 * Ejemplo: total=5, partes=2 produce tamanios [3,2] e inicios [0,3].
 * calcular_particion_2d_dimensiones lo invoca para N, M de A, M de B y P. */
static int preparar_eje(size_t total, int partes, EjeParticion2D *eje) {
    eje->partes = partes;
    /* columnas=1 convierte counts/displs en tamanios/inicios del eje. */
    return calcular_particiones(total, 1, partes, MAX_PROCESOS,
                                eje->tamanios, eje->inicios);
}

/* Helper de validacion de la etapa 1, despues de preparar los cuatro ejes.
 * "count" es el numero de elementos que recibira un bloque local en la futura
 * interfaz MPI; "valido" significa que ese numero cabe en un int.
 * No valida el contenido numerico de la matriz. El bloque mayor combina
 * los mayores tamanios de filas y columnas, por lo que basta comprobarlo.
 * Ejemplo: 3 filas por 2 columnas implican un count de 6 elementos.
 * La division comprueba el limite sin calcular un producto que desborde. */
static int count_local_valido(const EjeParticion2D *filas,
                             const EjeParticion2D *columnas) {
    /* El primer bloque es el mayor en una particion balanceada. */
    size_t max_filas = (size_t)filas->tamanios[0];
    size_t max_columnas = (size_t)columnas->tamanios[0];
    return max_columnas == 0 || max_filas <= (size_t)INT_MAX / max_columnas;
}

/* Entrada del proyecto: toma las dimensiones directamente de config.h.
 * Se llama "calcular_particion_2d" porque prepara el reparto de las matrices
 * configuradas; la grilla sigue siendo un dato de runtime para MPI.
 * No duplica la logica: delega en la variante con dimensiones explicitas. */
int calcular_particion_2d(int filas_grilla, int columnas_grilla,
                         Particion2D *particion) {
    return calcular_particion_2d_dimensiones((size_t)N, (size_t)M, (size_t)P,
                                            filas_grilla, columnas_grilla,
                                            particion);
}

/* Variante publica reutilizable: construye el plan de propiedad de A, B y C.
 * El sufijo "dimensiones" indica que recibe los tamanios en vez de leerlos
 * de config.h. Combina cortes de filas y columnas
 * para describir el reparto 2D; no ejecuta la distribucion de los valores.
 * Orden: validar entradas -> preparar ejes -> validar counts -> fusionar M.
 * M necesita dos particiones: columnas de A y filas de B. Fusionar sus cortes
 * garantiza que cada segmento pertenezca a un unico propietario de cada lado.
 * Ejemplo: cortes de A [0,2,4,6] y B [0,3,6] dan [0,2,3,4,6].
 * La futura coreografia SUMMA usara cada segmento para emparejar operandos,
 * traduciendo las coordenadas propietarias a ranks mediante MPI. */
int calcular_particion_2d_dimensiones(size_t n, size_t m, size_t p,
                                     int filas_grilla, int columnas_grilla,
                                     Particion2D *particion) {
    if (particion == NULL) return PARTICIONES_ARGUMENTOS_INVALIDOS;
    particion->num_segmentos = 0;
    if (n == 0 || m == 0 || p == 0 || filas_grilla <= 0 || columnas_grilla <= 0) {
        return PARTICIONES_ARGUMENTOS_INVALIDOS;
    }
    /* Comprobar la grilla sin multiplicar enteros que podrian desbordar. */
    if (filas_grilla > MAX_PROCESOS / columnas_grilla) {
        return PARTICIONES_CAPACIDAD_INSUFICIENTE;
    }
    if (n > INT_MAX || m > INT_MAX || p > INT_MAX ||
        n > SIZE_MAX / m || m > SIZE_MAX / p || n > SIZE_MAX / p) {
        return PARTICIONES_DESBORDAMIENTO;
    }

    particion->n = n;
    particion->m = m;
    particion->p = p;
    int estado = preparar_eje(n, filas_grilla, &particion->filas_a);
    if (estado != PARTICIONES_EXITO) return estado;
    estado = preparar_eje(m, columnas_grilla, &particion->columnas_a);
    if (estado != PARTICIONES_EXITO) return estado;
    estado = preparar_eje(m, filas_grilla, &particion->filas_b);
    if (estado != PARTICIONES_EXITO) return estado;
    estado = preparar_eje(p, columnas_grilla, &particion->columnas_b);
    if (estado != PARTICIONES_EXITO) return estado;

    if (!count_local_valido(&particion->filas_a, &particion->columnas_a) ||
        !count_local_valido(&particion->filas_b, &particion->columnas_b) ||
        !count_local_valido(&particion->filas_a, &particion->columnas_b)) {
        return PARTICIONES_DESBORDAMIENTO;
    }

    /* Fusion incremental: avanzar hasta el proximo corte de A o de B.
     * Se comparan indices de datos, nunca ranks ni vecinos de red. */
    int col_a = 0, fila_b = 0, cantidad = 0;
    size_t inicio = 0;
    while (inicio < m) {
        const EjeParticion2D *a = &particion->columnas_a;
        const EjeParticion2D *b = &particion->filas_b;
        while (col_a + 1 < a->partes && (size_t)a->inicios[col_a + 1] <= inicio) {
            col_a++;
        }
        while (fila_b + 1 < b->partes && (size_t)b->inicios[fila_b + 1] <= inicio) {
            fila_b++;
        }
        size_t fin_a = (size_t)a->inicios[col_a] + (size_t)a->tamanios[col_a];
        size_t fin_b = (size_t)b->inicios[fila_b] + (size_t)b->tamanios[fila_b];
        size_t fin = fin_a < fin_b ? fin_a : fin_b;
        if (cantidad >= MAX_PROCESOS) return PARTICIONES_CAPACIDAD_INSUFICIENTE;
        particion->segmentos[cantidad++] = (Segmento2D){
            .inicio = inicio, .ancho = fin - inicio,
            .propietario_columna_a = col_a, .propietario_fila_b = fila_b,
            .offset_a = inicio - (size_t)a->inicios[col_a],
            .offset_b = inicio - (size_t)b->inicios[fila_b]
        };
        inicio = fin;
    }
    particion->num_segmentos = cantidad;
    return PARTICIONES_EXITO;
}

/* Helper de consulta: combina una parte del eje de filas y otra de columnas.
 * "describir_bloque" indica que devuelve el rectangulo (inicio y tamanio),
 * sin copiar sus elementos ni reservar su almacenamiento. Recibe indices
 * ya validados por obtener_bloques_2d.
 * Ejemplo: filas [0,2), columnas [2,4) describen A01 de la matriz 4x4.
 * En la etapa 2 ese rectangulo servira para crear un datatype de subarray. */
static Bloque2D describir_bloque(const EjeParticion2D *filas, int fila,
                               const EjeParticion2D *columnas, int columna) {
    return (Bloque2D){
        .filas = (size_t)filas->tamanios[fila],
        .columnas = (size_t)columnas->tamanios[columna],
        .inicio_fila = (size_t)filas->inicios[fila],
        .inicio_columna = (size_t)columnas->inicios[columna]
    };
}

/* Segunda entrada publica: consulta un plan calculado con exito.
 * Se llama "obtener_bloques_2d" porque recupera los tres descriptores que
 * corresponden a una coordenada, sin recalcular el plan ni recibir matrices.
 * A usa filas de N/columnas de M; B usa filas de M/columnas de P;
 * C combina las filas de A con las columnas de B.
 * En la etapa 2, las coordenadas vendran de MPI_Cart_coords y los bloques
 * indicaran que enviar, recibir y reconstruir. En la etapa 3, el bloque C
 * define la region de resultado que acumulara cada proceso.
 * Ejemplo: consultar (0,1) de una grilla 2x2 devuelve A01, B01 y C01. */
int obtener_bloques_2d(const Particion2D *particion,
                      int fila_grilla, int columna_grilla,
                      BloquesProceso2D *bloques) {
    if (particion == NULL || bloques == NULL || particion->num_segmentos == 0 ||
        fila_grilla < 0 || columna_grilla < 0 ||
        fila_grilla >= particion->filas_a.partes ||
        columna_grilla >= particion->columnas_b.partes) {
        return PARTICIONES_ARGUMENTOS_INVALIDOS;
    }
    *bloques = (BloquesProceso2D){
        .a = describir_bloque(&particion->filas_a, fila_grilla,
                             &particion->columnas_a, columna_grilla),
        .b = describir_bloque(&particion->filas_b, fila_grilla,
                             &particion->columnas_b, columna_grilla),
        .c = describir_bloque(&particion->filas_a, fila_grilla,
                             &particion->columnas_b, columna_grilla)
    };
    return PARTICIONES_EXITO;
}

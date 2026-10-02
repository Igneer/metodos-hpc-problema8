#include "../src/mpi/2d/particion_2d.h"
#include "../src/mpi/2d/kernel_local.h"

#include <assert.h>
#include <limits.h>
#include <stdio.h>
#include <string.h>

/* Un plan de capacidad fija, reutilizado sin malloc ni VLA. */
static Particion2D particion;

/* La entrada principal debe usar la configuracion de compilacion, no valores
 * del ejemplo. Esta prueba tambien se ejecuta recompilando con -DN/-DM/-DP. */
static void probar_configuracion(void) {
    assert(calcular_particion_2d(1, 1, &particion) == PARTICIONES_EXITO);
    assert(particion.n == (size_t)N && particion.m == (size_t)M);
    assert(particion.p == (size_t)P);
    assert(particion.num_segmentos == 1);
    assert(particion.segmentos[0].inicio == 0);
    assert(particion.segmentos[0].ancho == (size_t)M);
    BloquesProceso2D bloques;
    assert(obtener_bloques_2d(&particion, 0, 0, &bloques) == PARTICIONES_EXITO);
    assert(bloques.a.filas == (size_t)N && bloques.a.columnas == (size_t)M);
    assert(bloques.b.filas == (size_t)M && bloques.b.columnas == (size_t)P);
    assert(bloques.c.filas == (size_t)N && bloques.c.columnas == (size_t)P);
    assert(calcular_particion_2d(0, 1, &particion) == PARTICIONES_ARGUMENTOS_INVALIDOS);
    assert(calcular_particion_2d(1, 1, NULL) == PARTICIONES_ARGUMENTOS_INVALIDOS);
}

/* Prueba de la estructura (etapa 1), no de la multiplicacion.
 * El nombre identifica el caso 4x4/grilla 2x2 del ejemplo pedagogico:
 * verifica los segmentos uniformes y la descripcion de A01. */
static void probar_ejemplo_cuadrado(void) {
    assert(calcular_particion_2d_dimensiones(4, 4, 4, 2, 2, &particion) == PARTICIONES_EXITO);
    assert(particion.num_segmentos == 2);
    for (int s = 0; s < 2; s++) {
        const Segmento2D *seg = &particion.segmentos[s];
        assert(seg->inicio == (size_t)(2 * s) && seg->ancho == 2);
        assert(seg->propietario_columna_a == s && seg->propietario_fila_b == s);
        assert(seg->offset_a == 0 && seg->offset_b == 0);
    }
    BloquesProceso2D bloques;
    assert(obtener_bloques_2d(&particion, 0, 1, &bloques) == PARTICIONES_EXITO);
    assert(bloques.a.filas == 2 && bloques.a.columnas == 2);
    assert(bloques.a.inicio_fila == 0 && bloques.a.inicio_columna == 2);
    /* Comprobar la seleccion de A01 contra valores conocidos, sin transferir. */
    static const real_t a[] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    static const real_t esperado[] = {3,4,7,8};
    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 2; j++) {
            assert(a[(bloques.a.inicio_fila + i) * 4 +
                     bloques.a.inicio_columna + j] == esperado[i * 2 + j]);
        }
    }
}

/* "Cortes desiguales" son las fronteras distintas de columnas de A y filas
 * de B. Esta prueba comprueba que se emparejen antes de ejecutar SUMMA:
 * para M=6/grilla 2x3 los anchos comunes deben ser 2,1,1,2. */
static void probar_cortes_desiguales(void) {
    assert(calcular_particion_2d_dimensiones(5, 6, 7, 2, 3, &particion) == PARTICIONES_EXITO);
    static const size_t inicios[] = {0, 2, 3, 4}, anchos[] = {2, 1, 1, 2};
    static const int columnas[] = {0, 1, 1, 2}, filas[] = {0, 0, 1, 1};
    static const size_t offsets_a[] = {0, 0, 1, 0}, offsets_b[] = {0, 2, 0, 1};
    assert(particion.num_segmentos == 4);
    for (int s = 0; s < 4; s++) {
        const Segmento2D *seg = &particion.segmentos[s];
        assert(seg->inicio == inicios[s] && seg->ancho == anchos[s]);
        assert(seg->propietario_columna_a == columnas[s]);
        assert(seg->propietario_fila_b == filas[s]);
        assert(seg->offset_a == offsets_a[s] && seg->offset_b == offsets_b[s]);
    }
    BloquesProceso2D bloques;
    assert(obtener_bloques_2d(&particion, 1, 2, &bloques) == PARTICIONES_EXITO);
    assert(bloques.a.filas == 2 && bloques.a.columnas == 2);
    assert(bloques.a.inicio_fila == 3 && bloques.a.inicio_columna == 4);
    assert(bloques.b.filas == 3 && bloques.b.columnas == 2);
    assert(bloques.b.inicio_fila == 3 && bloques.b.inicio_columna == 5);
    assert(bloques.c.filas == 2 && bloques.c.columnas == 2);
    assert(bloques.c.inicio_fila == 3 && bloques.c.inicio_columna == 5);
}

/* Helper exclusivo de pruebas: "marcar" registra las celdas de un bloque
 * en un mapa de cobertura. Una segunda marca detecta un solapamiento.
 * No distribuye datos; permite comprobar independientemente los descriptores. */
static void marcar_bloque(unsigned *cobertura, Bloque2D bloque,
                          size_t filas, size_t columnas) {
    assert(bloque.inicio_fila + bloque.filas <= filas);
    assert(bloque.inicio_columna + bloque.columnas <= columnas);
    for (size_t i = 0; i < bloque.filas; i++) {
        for (size_t j = 0; j < bloque.columnas; j++) {
            size_t posicion = (bloque.inicio_fila + i) * columnas +
                              bloque.inicio_columna + j;
            assert(++cobertura[posicion] == 1); /* Sin solapamientos. */
        }
    }
}

/* "Cobertura" significa que ninguna celda queda sin propietario ni tiene dos.
 * Barrido de casos pequenos: verifica los bloques de A/B/C con marcar_bloque
 * y comprueba que los segmentos recorran M una sola vez, dentro de sus duenos.
 * Respalda la etapa 1 antes de crear transferencias o broadcasts MPI. */
static void probar_cobertura(void) {
    static unsigned cobertura[3][25];
    for (size_t n = 1; n <= 5; n++) {
        for (size_t m = 1; m <= 5; m++) {
            for (size_t p = 1; p <= 5; p++) {
                for (int pr = 1; pr <= 4; pr++) {
                    for (int pc = 1; pc <= 4; pc++) {
                        if (pr > MAX_PROCESOS / pc) continue;
                        assert(calcular_particion_2d_dimensiones(n, m, p, pr, pc,
                                                                &particion) == PARTICIONES_EXITO);
                        memset(cobertura, 0, sizeof(cobertura));
                        for (int i = 0; i < pr; i++) {
                            for (int j = 0; j < pc; j++) {
                                BloquesProceso2D bloques;
                                assert(obtener_bloques_2d(&particion, i, j,
                                                         &bloques) == PARTICIONES_EXITO);
                                marcar_bloque(cobertura[0], bloques.a, n, m);
                                marcar_bloque(cobertura[1], bloques.b, m, p);
                                marcar_bloque(cobertura[2], bloques.c, n, p);
                            }
                        }
                        const size_t totales[] = {n * m, m * p, n * p};
                        for (int matriz = 0; matriz < 3; matriz++) {
                            for (size_t k = 0; k < totales[matriz]; k++) {
                                assert(cobertura[matriz][k] == 1);
                            }
                        }
                        size_t fin = 0;
                        for (int s = 0; s < particion.num_segmentos; s++) {
                            const Segmento2D *seg = &particion.segmentos[s];
                            int ca = seg->propietario_columna_a;
                            int fb = seg->propietario_fila_b;
                            assert(ca >= 0 && ca < pc && fb >= 0 && fb < pr);
                            assert(seg->inicio == fin && seg->ancho > 0);
                            assert(seg->offset_a + seg->ancho <=
                                   (size_t)particion.columnas_a.tamanios[ca]);
                            assert(seg->offset_b + seg->ancho <=
                                   (size_t)particion.filas_b.tamanios[fb]);
                            assert((size_t)particion.columnas_a.inicios[ca] +
                                   seg->offset_a == fin);
                            assert((size_t)particion.filas_b.inicios[fb] +
                                   seg->offset_b == fin);
                            fin += seg->ancho;
                        }
                        assert(fin == m);
                    }
                }
            }
        }
    }
}

/* El nombre distingue entradas invalidas de fronteras validas, como INT_MAX.
 * Comprueba la seguridad del contrato de la etapa 1: rechazar planes que no
 * caben, invalidar planes fallidos y preservar salidas de consultas invalidas. */
static void probar_errores_y_limites(void) {
    assert(calcular_particion_2d_dimensiones(4, 4, 4, 2, 2, NULL) == PARTICIONES_ARGUMENTOS_INVALIDOS);
    assert(calcular_particion_2d_dimensiones(0, 4, 4, 2, 2, &particion) == PARTICIONES_ARGUMENTOS_INVALIDOS);
    assert(calcular_particion_2d_dimensiones(4, 0, 4, 2, 2, &particion) == PARTICIONES_ARGUMENTOS_INVALIDOS);
    assert(calcular_particion_2d_dimensiones(4, 4, 0, 2, 2, &particion) == PARTICIONES_ARGUMENTOS_INVALIDOS);
    assert(calcular_particion_2d_dimensiones(4, 4, 4, 0, 2, &particion) == PARTICIONES_ARGUMENTOS_INVALIDOS);
    assert(calcular_particion_2d_dimensiones(4, 4, 4, 2, -1, &particion) == PARTICIONES_ARGUMENTOS_INVALIDOS);
    assert(calcular_particion_2d_dimensiones(4, 4, 4, MAX_PROCESOS, 2,
                                &particion) == PARTICIONES_CAPACIDAD_INSUFICIENTE);
    assert(calcular_particion_2d_dimensiones((size_t)INT_MAX + 1, 1, 1, 1, 1,
                                &particion) == PARTICIONES_DESBORDAMIENTO);
    assert(calcular_particion_2d_dimensiones(INT_MAX, 2, 1, 1, 1,
                                &particion) == PARTICIONES_DESBORDAMIENTO);
    assert(calcular_particion_2d_dimensiones(1, INT_MAX, 2, 1, 1,
                                &particion) == PARTICIONES_DESBORDAMIENTO);
    assert(calcular_particion_2d_dimensiones(INT_MAX, 1, 2, 1, 1,
                                &particion) == PARTICIONES_DESBORDAMIENTO);
    assert(particion.num_segmentos == 0);
    BloquesProceso2D bloques;
    assert(obtener_bloques_2d(&particion, 0, 0, &bloques) == PARTICIONES_ARGUMENTOS_INVALIDOS);
    assert(calcular_particion_2d_dimensiones(INT_MAX, 1, 1, 1, 1,
                                &particion) == PARTICIONES_EXITO);
    assert(obtener_bloques_2d(&particion, 0, 0, &bloques) == PARTICIONES_EXITO);
    assert(bloques.a.filas == INT_MAX && bloques.a.columnas == 1);
    assert(calcular_particion_2d_dimensiones(1, 1, 1, MAX_PROCESOS, 1,
                                &particion) == PARTICIONES_EXITO);
    assert(particion.num_segmentos == 1); /* Cortes repetidos no crean etapas vacias. */
    bloques.a.filas = 99;
    assert(obtener_bloques_2d(&particion, -1, 0, &bloques) == PARTICIONES_ARGUMENTOS_INVALIDOS);
    assert(bloques.a.filas == 99); /* Una consulta invalida no modifica la salida. */
    assert(obtener_bloques_2d(&particion, MAX_PROCESOS, 0,
                              &bloques) == PARTICIONES_ARGUMENTOS_INVALIDOS);
    assert(obtener_bloques_2d(&particion, 0, 1, &bloques) == PARTICIONES_ARGUMENTOS_INVALIDOS);
    assert(obtener_bloques_2d(NULL, 0, 0, &bloques) == PARTICIONES_ARGUMENTOS_INVALIDOS);
    assert(obtener_bloques_2d(&particion, 0, 0, NULL) == PARTICIONES_ARGUMENTOS_INVALIDOS);
}

/* Punto de entrada del ejecutable de pruebas, no del futuro matmul.
 * Ejecuta las comprobaciones de la etapa 1 sin inicializar MPI ni OpenBLAS. */
int main(void) {
    probar_configuracion();
    probar_ejemplo_cuadrado();
    probar_cortes_desiguales();
    probar_cobertura();
    probar_errores_y_limites();
    puts("Particion 2D: pruebas OK");
    return 0;
}

/*
 * matmul_ikj.c
 *
 * Multiplicacion secuencial C = A * B (A: N x M, B: M x P, C: N x P)
 * usando el orden de loops ikj, optimizado para localidad de cache.
 * Esta es la version que se usa como T_seq de referencia para
 * calcular speedup y eficiencia de las versiones paralelas (MPI/
 * OpenMP) del Proyecto 8.
 *
 * N, M, P son constantes de compilacion (ver src/common/config.h),
 * no parametros de linea de comandos: las matrices son estaticas
 * (viven en BSS, no en el heap), asi que el tamanio tiene que
 * conocerse al compilar. Para otro tamanio, recompilar con
 * -DN=... -DM=... -DP=... (el Makefile lo expone como SIZE=...).
 *
 * Uso:
 *   matmul_ikj [--semilla S] [--repeticiones R] [--csv]
 *
 * Ejemplos:
 *   ./matmul_ikj --repeticiones 10
 *   ./matmul_ikj --csv > resultados_ikj.csv
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../common/cli.h"
#include "../common/config.h"
#include "../common/matrix.h"
#include "../common/timing.h"

/* Matrices estaticas: tamanio fijo conocido en tiempo de compilacion,
 * reservadas en el segmento estatico del programa (no malloc/free,
 * no VLA en el stack). */
static real_t A[(size_t)N * M];
static real_t B[(size_t)M * P];
static real_t C[(size_t)N * P];

/* ---------------------------------------------------------------------
 * Multiplicacion ikj (orden optimizado para localidad de cache).
 *
 * Idea: fijar la fila i y el indice k, y barrer j en el loop mas
 * interno. Con matrices almacenadas en row-major (fila contigua en
 * memoria), eso hace que las DOS referencias del loop interno sean
 * secuenciales (stride 1):
 *
 *   b[k*p + j]  -> recorre la fila k de B de corrido
 *   c[i*p + j]  -> recorre la fila i de C de corrido
 *
 * a[i*m + k] es un escalar (a_ik) que se saca del loop interno, asi
 * que A se lee una vez por cada (i,k), no por cada (i,k,j).
 *
 * Beneficios concretos:
 *   1. Localidad espacial: cada linea de cache traida de B/C se
 *      aprovecha por completo (todos sus elementos se usan en
 *      iteraciones consecutivas de j), en vez de traerse y usarse un
 *      solo elemento como en ijk.
 *   2. Prefetching por hardware: un acceso secuencial y predecible es
 *      justo el patron que el prefetcher de la CPU anticipa mejor.
 *   3. Auto-vectorizacion: el compilador puede vectorizar el loop en
 *      j con SIMD (SSE/AVX) porque a_ik es invariante en j y ambos
 *      arreglos se leen/escriben con stride 1 y sin aliasing (gracias
 *      a `restrict`). Con -O3 -march=native, gcc genera codigo FMA
 *      vectorizado para este loop (confirmable con -fopt-info-vec).
 *   4. No cambia la complejidad (sigue siendo O(n*m*p)), solo mejora
 *      la constante oculta por el uso efectivo de la jerarquia de
 *      memoria.
 *
 * Nota: c debe venir inicializada en cero (se usa como acumulador).
 * ------------------------------------------------------------------- */
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

int main(int argc, char **argv) {
    char info_tamanio[64];
    snprintf(info_tamanio, sizeof(info_tamanio), "N=%d M=%d P=%d", N, M, P);

    ConfiguracionBenchmark config;
    if (!cli_parsear_argumentos(argc, argv, &config, info_tamanio)) {
        return 1;
    }

    matriz_llenar_aleatoria(A, N, M, config.semilla);
    matriz_llenar_aleatoria(B, M, P, config.semilla + 1);

    double flops = 2.0 * (double)N * (double)M * (double)P;

    if (config.csv) {
        printf("n,m,p,algoritmo,repeticion,segundos,gflops\n");
    } else {
        printf("N=%d M=%d P=%d algoritmo=ikj repeticiones=%d\n",
               N, M, P, config.repeticiones);
    }

    /* Cada repeticion corre la multiplicacion completa y se imprime
     * como dato crudo (una linea por corrida). La agregacion
     * estadistica (mediana, desvio) sobre estas corridas es
     * responsabilidad de un script externo de analisis (fase de
     * diseno experimental), no de este binario. */
    for (int r = 0; r < config.repeticiones; r++) {
        matriz_poner_cero(C, N, P);
        double t0 = ahora_segundos();
        multiplicar_ikj(A, B, C, N, M, P);
        double t1 = ahora_segundos();
        double transcurrido = t1 - t0;
        double gflops = flops / transcurrido / 1e9;

        if (config.csv) {
            printf("%d,%d,%d,ikj,%d,%.9f,%.4f\n",
                   N, M, P, r, transcurrido, gflops);
        } else {
            printf("repeticion %d: %.6f s (%.3f GFLOPS)\n", r, transcurrido, gflops);
        }
    }

    return 0;
}

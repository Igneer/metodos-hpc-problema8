# ScaLAPACK en `src/ref/matmul_pdgemm.c`: funciones, algoritmo SUMMA y paso que realiza cada una

## Fuentes consultadas

Fuentes primarias, leídas directamente (código fuente oficial de ScaLAPACK, rama `master`, y papers):

- **[S1]** PDGEMM: [`PBLAS/SRC/pdgemm_.c`](https://github.com/Reference-ScaLAPACK/scalapack/blob/master/PBLAS/SRC/pdgemm_.c)
- **[S2]** Variante AB: [`PBLAS/SRC/PTOOLS/PB_CpgemmAB.c`](https://github.com/Reference-ScaLAPACK/scalapack/blob/master/PBLAS/SRC/PTOOLS/PB_CpgemmAB.c)
- **[S3]** Difusión de paneles: [`PB_CInV2.c`](https://github.com/Reference-ScaLAPACK/scalapack/blob/master/PBLAS/SRC/PTOOLS/PB_CInV2.c)
- **[S4]** Tamaño de panel: [`PBLAS/SRC/pilaenv.f`](https://github.com/Reference-ScaLAPACK/scalapack/blob/master/PBLAS/SRC/pilaenv.f)
- **[S5]** [`TOOLS/descinit.f`](https://github.com/Reference-ScaLAPACK/scalapack/blob/master/TOOLS/descinit.f) y **[S6]** [`TOOLS/numroc.f`](https://github.com/Reference-ScaLAPACK/scalapack/blob/master/TOOLS/numroc.f)
- **[S7]** Redistribución: [`REDIST/SRC/pigemr.c`](https://github.com/Reference-ScaLAPACK/scalapack/blob/master/REDIST/SRC/pigemr.c) (documenta PIGEMR2D, la variante entera; PDGEMR2D es su variante `double` en `pdgemr.c`).
- **[S8]** BLACS: [Quick Reference](https://www.netlib.org/blacs/BLACS/QRef.html)
- **[S9]** Distribución block-cyclic: [ScaLAPACK Users' Guide, node76](https://www.netlib.org/scalapack/slug/node76.html)
- **[P1]** R. van de Geijn y J. Watts, *SUMMA: Scalable Universal Matrix Multiplication Algorithm*, Concurrency: Practice and Experience 9(4):255-274, 1997 ([LAWN 96](https://www.netlib.org/lapack/lawnspdf/lawn96.pdf)).
- **[P2]** S. Hunold y T. Rauber, *Automatic Tuning of PDGEMM towards Optimal Performance*, 2005 ([PDF](https://mescal.imag.fr/membres/sascha.hunold/paper/pdgemm_tuning_sahu_2005.pdf)). Cita a J. Choi, *A New Parallel Matrix Multiplication Algorithm on Distributed-Memory Concurrent Computers* (DIMMA), Concurrency: Practice and Experience 10:655-670, 1998, que no leí directamente.


## El algoritmo SUMMA

Según [P1], con A, B y C distribuidas en una malla de procesos r × c, C = A·B se calcula como una suma de actualizaciones de rango pequeño. Dado un ancho de panel `nb`, en cada iteración:

1. El proceso (o la columna de procesos) que posee las columnas actuales de A las **difunde dentro de su fila** de la malla.
2. El que posee las filas actuales de B las **difunde dentro de su columna**.
3. Cada proceso hace la **actualización local** `C_local += A_panel × B_panel`.

El paper lo describe así: los términos del costo "son debidos a la actualización de rango uno, la difusión dentro de la fila y la difusión dentro de la columna" ([P1], sección 5.1 *Basic Parallel Algorithm*; sección 5.2 *Pipelined Algorithm* para difusión en anillo). [P2] lo resume: *"In SUMMA, the processors P0 and P3 broadcast the first column of A along their row [...] the first row of matrix B [...] is broadcasted along the processor columns."* DIMMA mejora SUMMA enviando bloques de columnas (filas) en vez de una a la vez, para eliminar tiempos de espera ([P2]).

## Funciones usadas y paso que realiza cada una

| Función | Qué hace (según la fuente) | Paso en el programa / en SUMMA |
|---|---|---|
| `Cblacs_pinfo` | Da el identificador del proceso (0..NPROCS-1) y el número de procesos disponibles; se usa antes de configurar BLACS [S8]. | Inicialización. No es un paso de SUMMA. |
| `Cblacs_get` | Obtiene valores internos de BLACS; con `what=0` se usa para obtener el contexto de sistema por defecto [S8]. | Prepara la creación de las grillas. |
| `Cblacs_gridinit` | Mapea los procesos disponibles en una malla Pr × Pc ("R" = orden por filas); devuelve el contexto creado [S8]. | Crea la **malla 2D de procesos** de SUMMA. Se crean dos: la 2D y una 1x1 donde la raíz guarda las matrices completas. |
| `Cblacs_gridinfo` | Devuelve Pr, Pc y las coordenadas (fila, columna) del proceso [S8]. | Posición (i, j) de cada proceso en la malla; de ahí salen "mi fila" y "mi columna" para las difusiones. |
| `numroc_` | "Calcula el NÚmero de filas o columnas de una matriz distribuida que posee el proceso indicado" [S6]. | Tamaño del bloque local de A, B y C (cuánta memoria local se usa). |
| `descinit_` | Inicializa el descriptor (M, N, MB, NB, proceso origen, contexto, LLD) de una matriz distribuida [S5]. | Describe la **distribución block-cyclic** inicial de A, B y C [S9]. Precondición de SUMMA. |
| `pdgemr2d_` | "Copia una submatriz de A en una submatriz de B", con distintas grillas o tamaños de bloque. Quien no pertenece a una grilla debe poner `CTXT = -1` en su descriptor [S7]. | **No es parte de SUMMA.** Reparte A y B desde la raíz (1x1) a la malla 2D antes del producto y recoge C después. |
| `pdgemm_` | Calcula `sub(C) := alpha*op(sub(A))*op(sub(B)) + beta*sub(C)` sobre matrices distribuidas [S1]. | **Todo el algoritmo tipo SUMMA** (detalle abajo). |
| `Cblacs_gridexit` / `Cblacs_exit` | Liberan un contexto / todos los contextos y la memoria de BLACS [S8]. | Limpieza. |

## Qué ocurre dentro de `pdgemm_`

1. **Elección de variante** [S1]: según M, N, K y las distribuciones, estima el costo de comunicación de tres variantes, `PB_CpgemmAB`, `PB_CpgemmBC` y `PB_CpgemmAC`, y usa la mejor (el código comenta que AB es "both operands sub(A) and sub(B) are communicated (M, N >> K)"). La variante AB es la que corresponde al esquema de SUMMA descrito arriba. No se verificó cuál elige para este problema.
2. **Topología de difusión** [S1]: si M, N y K superan el tamaño lógico de bloque, fuerza una topología en anillo para las difusiones por fila y por columna (el equivalente del algoritmo *pipelined* de [P1]).
3. **Tamaño de panel** [S4]: `kb = PILAENV`, que en la versión de referencia vale **32** para `double`. Es el `nb` de SUMMA.
4. **Bucle `for (k = 0; k < K; k += kb)`** en `PB_CpgemmAB` [S2], por cada panel de ancho `kbb`:
   - **Empaquetado**: si las columnas de A (o filas de B) del panel no son contiguas en memoria, las copia a un buffer (`PB_CVMpack`). Es el empaquetado que [P1] también hace antes de enviar.
   - **Difusión del panel de A por la fila y del de B por la columna**: `PB_CInV2` ([S3]) usa `Cgebs2d`/`Cgebr2d` (envío/recepción de difusión BLACS) con alcance `ROW` para el panel de A y `COLUMN` para el de B, dejando el panel replicado en los buffers `WA` y `WB`.
   - **Actualización local**: llama a DGEMM, `gemm(..., &Cmp, &Cnq, &kbb, ALPHA, WA, ..., WB, ..., tbeta, Cptr, &Cld)` sobre la porción local de C.
5. Si A y B no abarcan más de una fila/columna de procesos, se omite todo el empaquetado y hay una sola llamada a DGEMM [S2].

## Row-major vs column-major

La memoria es una línea de posiciones; una matriz 2D hay que aplanarla. Ejemplo, matriz de 2 filas y 3 columnas (`1 2 3 / 4 5 6`):

| | Row-major (C) | Column-major (Fortran, ScaLAPACK) |
|---|---|---|
| Orden en memoria | `1 2 3 4 5 6` (fila tras fila) | `1 4 2 5 3 6` (columna tras columna) |
| Posición de (i, j) | `i*columnas + j` | `i + j*filas` |
| Contiguo | una fila | una columna |

**El problema.** ScaLAPACK/PBLAS interpreta toda la memoria como column-major (heredado de Fortran; no hay opción row-major). El proyecto guarda las matrices en row-major (`matrix.h`). Si se le pasaran `A` y `B` tal cual y se le pidiera `A·B`, leería otras matrices y calcularía otro producto.

**El artilugio, en dos pasos.**

1. *Los mismos números se pueden leer de dos formas.* La memoria `1 2 3 4 5 6` leída por filas (C) es `1 2 3 / 4 5 6`; leída por columnas (ScaLAPACK) es `1 4 / 2 5 / 3 6`, la transpuesta. No se movió ningún dato, solo cambió cómo se leen. Vale para A, B y C.
2. *La transpuesta de un producto invierte el orden:* (A·B)ᵀ = Bᵀ·Aᵀ.

**Cómo se combinan.** Lo que ScaLAPACK ve es Aᵀ, Bᵀ y Cᵀ. En lugar de pedirle C = A·B, se le pide lo equivalente en su mundo: multiplicar lo que ve como Bᵀ por lo que ve como Aᵀ y guardar el resultado en lo que ve como Cᵀ. ScaLAPACK escribe Cᵀ en column-major y, leída esa misma memoria en row-major, es C.

| Matriz | Dimensiones reales (row-major) | Dimensiones que se declaran (column-major) |
|---|---|---|
| A | N×M | M×N |
| B | M×P | P×M |
| C | N×P | P×N |

En la llamada a `pdgemm_` el primer operando es `B` y el segundo `A` (orden invertido), con `m=P`, `n=N`, `k=M`. Los únicos cambios respecto de una llamada normal son las dimensiones transpuestas y el orden inverso de los operandos; los datos nunca se tocan. El `LLD` de cada descriptor es la dimensión "de filas" que ve ScaLAPACK (M para A, P para B y para C).

**Dónde está en el código** (`src/ref/matmul_pdgemm.c`):

- Descriptores: `describir_global` y `describir_distribuida` se llaman con (M, N) para A, (P, M) para B y (P, N) para C.
- `pdgemr2d_` usa esas mismas dimensiones transpuestas.
- `pdgemm_("N", "N", &dim_m, &dim_n, &dim_k, ..., B_local, ..., A_local, ...)` con `dim_m = P`, `dim_n = N`, `dim_k = M`.

**Verificación de corrección.**: bash scripts/test_matriz_conocida.sh

# Plan de implementación: SUMMA 2D modular en C

**Secuencia principal: definir la estructura → construir el cimiento → ejecutar SUMMA.**

**Estado:** etapas 1, 2 y 3 implementadas y probadas; pendiente el trabajo complementario (OpenBLAS, mediciones). Las tareas y el punto de continuación están en [seguimiento_summa_2d.md](seguimiento_summa_2d.md).

La partición establece la propiedad de los datos; la infraestructura permite moverlos y operar sobre ellos; SUMMA organiza los productos que completan C. Distribuir en 2D no equivale a implementar SUMMA.

El ejemplo completo está en [explicación y ejemplos.md](explicaci%C3%B3n%20y%20ejemplos.md).

## Etapa 1 — Definir la estructura mediante la partición 2D

**Pregunta:** ¿qué posee cada proceso y qué resultado debe producir?

- Describir tamaños, posiciones y propietarios lógicos de los bloques de A, B y C.
- Definir segmentos de la dimensión compartida, incluyendo cortes desiguales y bloques vacíos.
- Establecer responsabilidades e interfaces de partición, topología, transferencias, SUMMA y kernel local, incorporando sus módulos cuando se implementen.
- Definir el contrato intercambiable `C_local += A × B`, con dimensiones y strides explícitos.

Reutilizar `calcular_particiones()` y las interfaces comunes existentes. Adaptar la planificación de segmentos del SUMMA actual sin modificar su archivo: sus funciones privadas `static` no son enlazables como biblioteca.

**Ejemplo:** con matrices 4×4 y grilla 2×2, P(0,0) posee A00 y B00, pero debe producir `C00 = A00 × B00 + A01 × B10`.

**Entregable:** descripción de bloques y segmentos, con contratos claros. Recibe dimensiones y coordenadas lógicas; no crea comunicadores, resuelve ranks ni mueve matrices.

Archivos: `src/mpi/2d/particion_2d.h`, `particion_2d.c`, `kernel_local.h` y la prueba `tests/test_particion_2d.c`. La entrada principal es `calcular_particion_2d(filas_grilla, columnas_grilla, &particion)` y utiliza N, M y P de `config.h`. La variante `calcular_particion_2d_dimensiones()` conserva las dimensiones explícitas para pruebas y reutilización. Desde la raíz, se puede verificar sin MPI ni OpenBLAS:

```sh
gcc -std=c11 -Wall -Wextra -Wpedantic tests/test_particion_2d.c src/mpi/2d/particion_2d.c src/common/particiones.c -o /tmp/test_particion_2d
/tmp/test_particion_2d
```

## Etapa 2 — Construir el cimiento

**Pregunta:** ¿cómo hacemos accesibles los datos y las operaciones definidos?

- Crear grilla y grupos de filas/columnas con `MPI_Dims_create`, `MPI_Cart_create` y `MPI_Cart_sub`.
- Consultar coordenadas y ranks con `MPI_Cart_coords` y `MPI_Cart_rank`, en el comunicador correspondiente, sin depender del orden mundial.
- Preparar buffers estáticos y datatypes con `MPI_Type_create_subarray` y `MPI_Type_commit`; representar bloques vacíos con count cero y un datatype básico válido.
- Implementar distribución inicial y reconstrucción mediante transferencias MPI, además de un kernel propio con strides.
- Establecer ownership de recursos y liberación mediante `MPI_Type_free` y `MPI_Comm_free`.

**Ejemplo:** MPI describe A01 dentro de A global y transfiere los valores 3, 4, 7 y 8 a un buffer local contiguo.

**Entregable:** infraestructura capaz de distribuir bloques, operar localmente y reconstruir matrices, sin ejecutar todavía la coreografía SUMMA.

## Etapa 3 — Ejecutar la coreografía de SUMMA

**Pregunta:** ¿en qué orden compartimos operandos y acumulamos resultados?

Inicializar C una vez; distribuir los bloques mediante el cimiento; para cada segmento, difundir A por filas y B por columnas con `MPI_Bcast`, e invocar el kernel para acumular. Al terminar, recolectar los bloques de C.

Los propietarios pueden utilizar vistas con stride y los receptores paneles compactos; los datatypes comunicados deben tener firmas compatibles. El orden de colectivas debe ser compatible en cada grupo.

**Ejemplo:** P(0,0) acumula primero A00×B00 y después A01×B10. C00 termina con las filas `[10,14]` y `[26,30]`.

**Entregable:** multiplicación completa mediante SUMMA. No requiere desplazamientos; si una operación futura necesita vecinos, se obtendrán con `MPI_Cart_shift`.

## Trabajo complementario para respaldar la implementación

### A. Pruebas de correctitud

Verificar particiones, distribución/reconstrucción, strides, resultados conocidos, matrices rectangulares, grillas no cuadradas, bloques vacíos y repeticiones. Comparar valores completos con referencia independiente y tolerancias absoluta/relativa, rechazando resultados no finitos.

Las pruebas se presentan separadas del plan principal, pero se ejecutan al construir cada parte, no únicamente al final.

### B. Integración opcional de OpenBLAS

Conocer la interfaz del compañero y agregar un adaptador nuevo al contrato del kernel. Utilizar GEMM row-major, strides correctos, `alpha = 1` y `beta = 1`. SUMMA no cambia y el backend propio sigue disponible sin OpenBLAS.

### C. Mediciones y conclusiones

Agregar scripts independientes de compilación/ejecución y conservar datos crudos. Medir distribución, núcleo y recolección con `MPI_Wtime` y máximos mediante `MPI_Reduce`. Comparar con 1D controlando entradas, backend e hilos; declarar las diferencias que no puedan igualarse sin modificar ejecutables existentes.

## Directivas comunes

- Agregar módulos nuevos y preservar todo el código existente, incluido el trabajo sobre OpenBLAS. Reutilizar `config`, `matrix`, `cli` y `particiones` mediante sus interfaces públicas.
- Mantener C, SUMMA, el contrato A de N×M por B de M×P, dimensiones de compilación, matrices estáticas row-major y `restrict` en kernels propios; no reservar matrices ocultamente en los módulos.
- Delegar a MPI la topología, identificación de procesos y regiones comunicadas. No calcular ranks o vecinos mediante fórmulas manuales; la aritmética de partición de datos sí corresponde al algoritmo.
- Distinguir tamaños activos de capacidad estática: admitir distintas grillas en runtime exige inicialmente buffers conservadores. No prometer reducción de capacidad por proceso que no se haya implementado y medido.
- Mantener OpenBLAS opcional y encapsulado; la restricción estática corresponde a las matrices de aplicación, no a la memoria interna de las bibliotecas.
- Explicar y verificar cada parte antes de avanzar. Documentar con ejemplos y sin LaTeX.

## Alternativa futura: delegar el producto distribuido a una biblioteca

**Propuesta opcional, no implementada ni elegida para sustituir el plan.** OpenBLAS resuelve un producto local; ScaLAPACK y SLATE pueden encargarse también de la comunicación del producto distribuido. No son backends intercambiables de `KernelLocal`: requieren un ejecutable o adaptador de nivel distribuido independiente.

Una llamada GEMM distribuida no carga los archivos ni convierte automáticamente el reparto actual. La aplicación todavía prepara matrices, distribución, buffers, configuración y validación. Tampoco demuestra que el equipo implementó la coreografía SUMMA, ni garantiza ser más rápida para los tamaños del proyecto.

### ¿Cuál vale la pena?

| Opción | Ventajas | Costos y alcance |
|--------|----------|------------------|
| ScaLAPACK/PBLAS | GEMM distribuida mediante `PDGEMM`, infraestructura consolidada en CPU, uso de BLAS como OpenBLAS. Buen candidato a referencia adicional. | Distribución 2D block-cyclic, descriptores y contexto BLACS; layout local column-major e interoperabilidad con interfaces Fortran. Integración de dificultad media, menor si el entorno ya está preparado. |
| SLATE | GEMM distribuida moderna, paralelismo por tareas y posibilidad de CPU/GPU; dispone de API C opcional. | Biblioteca interna C++17/OpenMP, BLAS++/LAPACK++, configuración y API C a habilitar. Mayor costo de integración; GPU amplía aún más el experimento. Conviene como trabajo futuro, salvo que ya esté instalada y haya experiencia en el equipo. |

**Recomendación:** para este proyecto, una PoC CPU de ScaLAPACK puede valer la pena como comparación industrial acotada. SLATE no es necesaria para demostrar la teoría y probablemente supera el esfuerzo razonable del alcance actual. Ninguna opción es simplemente enlazar una biblioteca y pasarle los tres arrays actuales.

### Plan mínimo con ScaLAPACK

1. Agregar un ejecutable nuevo, por ejemplo `src/mpi/matmul_scalapack_poc.c`, y un script de compilación independiente. Preparar ScaLAPACK con MPI y BLAS/LAPACK compatibles, comprobando ABI y tamaño de enteros; BLACS está integrado en la distribución de referencia moderna.
2. Establecer un contexto y grilla BLACS sobre los procesos MPI. Consultar coordenadas con las funciones de biblioteca y comprobar su correspondencia con el comunicador, sin inventar vecinos ni asumir ranks mundiales.
3. Usar `NUMROC` para tamaños locales y `DESCINIT` para describir A/B/C con distribución block-cyclic. Reservar buffers de aplicación estáticos de capacidad suficiente y adaptar los datos row-major a buffers column-major del ejecutable nuevo. El reparto de la etapa 1 no representa una distribución block-cyclic general y no se reutiliza directamente.
4. Distribuir las entradas mediante las rutinas de redistribución de la biblioteca, por ejemplo `PDGEMR2D`, con contextos/descriptores de origen y destino válidos; no escribir un reparto por elementos. Ejecutar `PDGEMM` en los procesos participantes, con `alpha=1`, `beta=0` para calcular C completo. La biblioteca organiza las comunicaciones durante el producto.
5. Recolectar mediante redistribución, convertir el layout para validar y medir tanto GEMM como tiempo total. Reutilizar las entradas y utilidades comunes, controlar hilos de OpenBLAS y cerrar recursos BLACS/MPI según sus interfaces.

### Si se elige SLATE

1. Construir o utilizar SLATE en CPU con API C habilitada (`c_api=yes`) y BLAS disponible; la aplicación puede seguir en C aunque la biblioteca se construya con C++/OpenMP. Verificar el soporte de hilos MPI requerido por la configuración elegida.
2. Agregar un ejecutable independiente, crear matrices distribuidas con las interfaces documentadas y adaptar los buffers/layout. Una ruta posible utiliza almacenamiento compatible con ScaLAPACK; no asumir que envolver los arrays globales ya distribuye su contenido. La biblioteca puede reservar metadata y espacios de trabajo internos.
3. Invocar GEMM distribuida mediante la API disponible en la versión instalada, configurar ejecución CPU, recuperar/validar resultados y comparar con hilos y criterios de timing declarados.

**Relación con las directivas:** estas rutas delegan la grilla y la comunicación a BLACS/SLATE; no garantizan reproducir internamente las llamadas cartesianas exigidas a la PoC propia. Tampoco conservan necesariamente su layout. Deben presentarse como una alternativa/comparación separada, no como cumplimiento automático de todas las restricciones de la implementación manual modular.

Fuentes oficiales: [ScaLAPACK](https://www.netlib.org/scalapack/), [PDGEMM](https://github.com/Reference-ScaLAPACK/scalapack/blob/master/PBLAS/SRC/pdgemm_.c), [DESCINIT](https://github.com/Reference-ScaLAPACK/scalapack/blob/master/TOOLS/descinit.f), [SLATE](https://icl.bitbucket.io/slate/) e [instalación y API C de SLATE](https://github.com/icl-utk-edu/slate/blob/master/INSTALL.md).

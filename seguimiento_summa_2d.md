# Seguimiento de SUMMA 2D

**Estado al 2026-10-02: etapas 1, 2 y 3 completadas y probadas; queda el trabajo complementario.**

Este documento permite retomar la implementación sin confundir tareas terminadas con propuestas. El [plan principal](plan_summa_2d.md) mantiene las tres etapas; [explicación y ejemplos.md](explicaci%C3%B3n%20y%20ejemplos.md) contiene el recorrido numérico.

## Tareas completadas

- [x] Definir el plan: estructura 2D, cimiento y coreografía SUMMA, con trabajo complementario separado.
- [x] Incorporar `src/mpi/2d/particion_2d.h` con descriptores de ejes, bloques, propietarios lógicos y segmentos.
- [x] Incorporar `src/mpi/2d/particion_2d.c`, reutilizando `calcular_particiones()` sin modificar su implementación.
- [x] Resolver cortes desiguales de la dimensión compartida mediante segmentos comunes, sin etapas vacías.
- [x] Contemplar bloques vacíos y rechazar argumentos, capacidades y tamaños incompatibles con los límites definidos.
- [x] Incorporar `src/mpi/2d/kernel_local.h`: contrato `C += A×B`, con dimensiones y strides; solo interfaz, sin backend implementado.
- [x] Incorporar `tests/test_particion_2d.c` con ejemplos conocidos y 2.000 combinaciones pequeñas de dimensiones/grillas.
- [x] Ejecutar la prueba 2D normal y con UBSan en modo trap, y la regresión de particiones 1D: resultados correctos.
- [x] Documentar el ejemplo completo y el comando para verificar la etapa 1.
- [x] Comentar cada función de partición y pruebas con su propósito, motivo del nombre y lugar en el flujo; explicar el rol de `KernelLocal` y volver a ejecutar la prueba 2D.
- [x] Hacer que `calcular_particion_2d()` tome N, M y P de `config.h`; conservar `calcular_particion_2d_dimensiones()` para pruebas y verificar las configuraciones 512×512 y A de 5×6 por B de 6×7.
- [x] Preservar el código existente del proyecto.

## Etapa 2 completada (cimiento)

- [x] `src/mpi/2d/topologia_2d.{h,c}`: grilla con `MPI_Dims_create`/`MPI_Cart_create`, grupos de fila y columna con `MPI_Cart_sub`, coordenadas y raíz vía MPI; `topologia_2d_liberar` hace `MPI_Comm_free`.
- [x] `src/mpi/2d/transferencias_2d.{h,c}`: datatypes `MPI_Type_create_subarray` (bloque vacío = count 0 con `MPI_DOUBLE`), `distribuir_2d` y `reconstruir_2d` para A, B y C; ownership con `tipo_bloque_2d_liberar`.
- [x] `src/mpi/2d/kernel_propio.{h,c}`: `kernel_propio_ikj`, compatible con `KernelLocal` (strides, acumulación, dimensiones cero).
- [x] `tests/test_infraestructura_2d.c` y `tests/probar_infraestructura_2d.sh`: ida y vuelta de A, B y C, coherencia de coordenadas/subcomunicadores, kernel con strides; 24 combinaciones (dimensiones 5x6x7, 4x4x4, 3x2x9, 1x1x1 con 1, 2, 3, 4, 6 y 8 procesos, incluidas grillas no cuadradas y bloques vacíos): todas OK.

Verificar (necesita `mpicc`; en este equipo se usó WSL): `sh tests/probar_infraestructura_2d.sh`.

## Etapa 3 completada (coreografía SUMMA)

- [x] `src/mpi/2d/summa_2d.{h,c}`: C se inicializa una vez; por segmento, `MPI_Bcast` de A por la fila y de B por la columna, y llamada al kernel. El dueño de A transmite una vista con `MPI_Type_vector` (sin copias) y los receptores usan un panel compacto; el dueño de B transmite directo desde su bloque.
- [x] `src/mpi/matmul_summa_2d.c`: ejecutable (mismo CLI y CSV que los demás; algoritmo `summa_2d`), con tiempos máximos de distribución, SUMMA y recolección vía `MPI_Reduce`.
- [x] `tests/test_summa_2d.c` y `tests/probar_summa_2d.sh`: comparación exacta con producto serial, 6 dimensiones × 9 cantidades de procesos (54 casos, grillas no cuadradas, cortes desiguales y bloques vacíos), todas OK. La salida de `--imprimir` coincide con la de `matmul_ikj` serial con 1, 4 y 6 procesos.

## Referencia con ScaLAPACK (hecha)

- [x] `src/ref/matmul_pdgemm.c`: C = A·B con `PDGEMM` (PBLAS, algoritmo tipo SUMMA). Grilla BLACS, distribución y recolección con `PDGEMR2D` (bloques cíclicos NB, 64 por defecto) y sin convertir layout: se calcula Cᵀ = Bᵀ·Aᵀ sobre la misma memoria row-major. Mismo CLI/CSV, algoritmo `pdgemm_2d` (el que ya espera `scripts/benchmark.py`). Verificado con matrices enteras contra `matmul_ikj` serial (NB 2, 3 y 64; 1, 2, 3, 4, 6 y 8 procesos). Se recomienda ejecutar con `OPENBLAS_NUM_THREADS=1` para no mezclar hilos de OpenBLAS con procesos MPI.
- [ ] Falta `src/ref/matmul_dgemm.c`, que `scripts/benchmark.py` y `scripts/probar_referencias.sh` también esperan y no existe en el repositorio.

## Pendiente: trabajo complementario

- [ ] Integrar el trabajo del compañero sobre OpenBLAS mediante un adaptador opcional; su interfaz concreta aún debe conocerse.
- [ ] Ampliar pruebas a comunicación, kernels, SUMMA y varias repeticiones, verificando cada parte al implementarla.
- [ ] Medir distribución, núcleo y recolección; conservar datos crudos y controlar backend/hilos en las comparaciones.
- [ ] Evaluar, solo si se decide retomarlo, la alternativa independiente ScaLAPACK/SLATE descrita al final del plan.

## Cómo verificar lo que ya funciona

Desde la raíz del repositorio, sin MPI ni OpenBLAS:

```sh
gcc -std=c11 -Wall -Wextra -Wpedantic tests/test_particion_2d.c src/mpi/2d/particion_2d.c src/common/particiones.c -o /tmp/test_particion_2d
/tmp/test_particion_2d
```

Salida esperada: `Particion 2D: pruebas OK`. La suite existente se ejecuta compilando `tests/test_particiones.c` junto con `src/common/particiones.c`.

**Notas de la verificación realizada:** ASan no pudo enlazarse por bibliotecas del entorno faltantes; UBSan funcionó en modo trap. Permanece un warning preexistente de comentario en `config.h:36`. La partición se probó sin procesos MPI porque aún no realiza comunicación.

## Límites para quien continúe

La etapa 1 entrega metadata, no una nueva multiplicación distribuida. El SUMMA previo sigue en `src/mpi/matmul_summa.c`; no debe confundirse con el nuevo desarrollo modular. Los futuros módulos y ejecutables se agregan sin modificar los existentes. Las matrices de aplicación siguen siendo estáticas; topología e identificación de procesos se delegan a MPI. Las funciones `static` de los ejecutables anteriores sirven como referencia, no como biblioteca enlazable.

Los archivos nuevos figuran sin seguimiento en Git en el momento de registrar este estado; no se creó un commit de esta etapa durante este trabajo. Antes de continuar, revisar `git status` y volver a ejecutar las pruebas. Actualizar este seguimiento solamente después de completar y verificar una tarea.

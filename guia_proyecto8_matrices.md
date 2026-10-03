# Guía de trabajo — Proyecto 8: Multiplicación de Matrices en Paralelo

## 1. Encuadre del problema

**Problema:** dadas dos matrices densas A (n×m) y B (m×p), calcular C = A·B, donde cada elemento es:

```
C[i][j] = Σ(k=0..m-1) A[i][k] * B[k][j]
```

**Por qué es un buen caso de estudio para paralelismo:**
- Complejidad O(n³) para matrices cuadradas de tamaño n → crece muy rápido, así que el costo computacional domina sobre el overhead de paralelizar (buen "grano" de trabajo).
- Cada elemento de C se puede calcular de forma independiente → paralelismo de datos casi perfecto en el nivel de la tarea.
- Es sensible al patrón de acceso a memoria (localidad de caché), lo cual te da un eje extra de optimización más allá de "repartir trabajo entre procesos".
- Es el ejemplo clásico para comparar estrategias de descomposición de dominio (por filas, por bloques, 2D) y para ver cómo el costo de comunicación cambia con la estrategia.

**Objetivos del trabajo (lo que se espera que puedan responder al final):**
1. ¿Cómo escala el tiempo de ejecución al aumentar el tamaño de la matriz (n) y el número de procesos (p)?
2. ¿Qué estrategia de distribución de carga da mejor rendimiento y por qué?
3. ¿Cuánto de la ganancia teórica (ideal) se logra en la práctica? (eficiencia paralela)
4. ¿Dónde está el cuello de botella cuando el rendimiento se degrada (comunicación, balance de carga, memoria)?

**Limitaciones a tener en cuenta y declarar explícitamente:**
- El tamaño máximo de matriz que corre en la RAM disponible de cada sistema de prueba.
- El número de núcleos/cores físicos reales de cada máquina (hyperthreading no es lo mismo que cores físicos).
- Ruido en las mediciones por otros procesos del sistema operativo → de ahí la necesidad de repetir corridas y usar mediana.
- Si usan MPI en una sola PC, la comunicación es memoria compartida simulada por sockets/loopback, lo cual no representa un clúster real con red física.

---

## 2. Resolución algorítmica

### 2.1 Algoritmo secuencial base (referencia)
Triple bucle clásico `for i, for j, for k`. Este es el que van a usar como baseline para el speedup absoluto.

Mejora simple sin paralelizar (para que la baseline no sea artificialmente mala):
- Reordenar los bucles como `i-k-j` en vez de `i-j-k`: mejora el acceso secuencial a memoria de B, porque se recorre por filas en vez de por columnas. Esto solo mejora localidad de caché, no cambia la complejidad.

### 2.2 Alternativas algorítmicas (para mencionar en el marco teórico, no necesariamente implementar)
- **Algoritmo de bloques (blocked/tiled multiplication):** divide las matrices en submatrices que caben en caché, reduce cache misses.
- **Algoritmo de Strassen:** O(n^2.807), reduce multiplicaciones a costa de más sumas y complejidad de implementación. Interesante para la sección de "trabajo futuro" pero no es obligatorio implementarlo.
- **Cannon's Algorithm / SUMMA:** algoritmos específicamente diseñados para paralelismo distribuido en grilla 2D de procesos, minimizan comunicación. Son la referencia "seria" en HPC para este problema.

### 2.3 Recomendación práctica para el TP
Para el nivel de este parcial, alcanza con:
1. Implementar el algoritmo secuencial (baseline).
2. Implementar 1 o 2 estrategias de paralelización (ver sección 3).
3. Opcionalmente, agregar blocking/tiling como optimización de memoria dentro de la versión paralela, y medir el efecto por separado.

---

## 3. Estrategias de paralelización

### 3.1 Elección de modelo
- **Memoria compartida (OpenMP / threads):** más simple, buena para una sola máquina multi-core. Ideal si van a correr en "PC".
- **Memoria distribuida (MPI):** necesaria si van a correr en "servidor" con varios nodos, o si quieren mostrar escalado más allá de los cores de una PC.
- **Híbrido (MPI + OpenMP):** un proceso MPI por nodo, threads OpenMP dentro de cada nodo. Da para un análisis más rico si tienen tiempo, pero no es obligatorio.

Elijan el modelo según qué hardware van a usar realmente (revisen la consigna: "PC, servidor, etc." → probablemente MPI les da más flexibilidad para correr en ambos con el mismo código, variando `-np`).

### 3.2 Estrategias de distribución de carga (esto es el corazón del análisis pedido)

**a) Distribución por filas (row-wise / 1D)**
- Se reparten las filas de A entre los p procesos (cada proceso recibe ⌈n/p⌉ filas de A, y una copia completa de B).
- Cada proceso calcula un bloque de filas de C de forma independiente.
- Ventaja: muy simple de implementar, poco código de comunicación (broadcast de B, scatter de A, gather de C).
- Desventaja: cada proceso necesita toda la matriz B en memoria → no escala bien en memoria para matrices muy grandes; poca localidad si m es grande.

**b) Distribución por bloques 2D (block/grid decomposition)**
- Se organiza a los procesos en una grilla √p × √p (necesita p cuadrado perfecto, o adaptarlo).
- Cada proceso recibe un bloque de A y un bloque de B, y calcula un bloque de C.
- Requiere más comunicación (cada proceso necesita bloques de otros procesos a medida que avanza — algoritmo de Cannon o SUMMA).
- Ventaja: mejor uso de memoria por proceso, mejor escalabilidad para matrices grandes y muchos procesos.
- Desventaja: mucho más complejo de implementar correctamente.

**c) Distribución cíclica / por bloques de columnas**
- Variante de (a), útil si quieren mostrar un tercer punto de comparación con poco esfuerzo extra (cambiar el eje de partición).

**Recomendación:** implementen (a) como estrategia principal (row-wise), y si el tiempo lo permite, agreguen (b) o al menos una variante de distribución cíclica para poder comparar "estrategia de distribución de carga" como pide la consigna. Con solo (a) ya se puede responder gran parte de las preguntas, pero comparar 2 estrategias enriquece mucho la discusión y es explícitamente pedido ("explore alternativas variando… la estrategia de distribución de la carga").

### 3.3 Balance de carga cuando n no es divisible por p
No asuman que n es siempre múltiplo de p. Definan una repartición que le dé filas de más a los primeros `n mod p` procesos (patrón clásico de "distribución balanceada"), y midan si esto genera desbalance quenotable en los tiempos por proceso (pueden loguear el tiempo de cómputo de cada rank individualmente).

### 3.4 Comunicación
- Punto crítico: minimizar comunicaciones bloqueantes y evitar broadcast innecesario en cada iteración.
- Para (a): un solo `Bcast` de B, un `Scatterv` de A, un `Gatherv` de C al final. Eviten enviar datos elemento por elemento.
- Documenten el tiempo de comunicación vs cómputo por separado si pueden (instrumentar con `MPI_Wtime()` alrededor de cada fase), esto ayuda mucho a explicar por qué la eficiencia cae con p grande.

---

## 4. Diseño experimental

### 4.1 Generación de datos
- Generar matrices con valores aleatorios (ej. uniforme en [0,1) o enteros pequeños) usando una semilla fija para reproducibilidad.
- Probar al menos 3-4 tamaños de matriz representativos, por ejemplo: n = 256, 512, 1024, 2048 (ajusten según la RAM/tiempo disponible; matrices muy grandes con el algoritmo O(n³) secuencial pueden tardar mucho).
- Si quieren estudiar "distintos tipos de pruebas", pueden variar también:
  - Matrices cuadradas vs rectangulares.
  - Datos densos "normales" vs matrices dispersas (aunque el algoritmo denso no aprovecha la dispersión, es útil para comentar limitaciones).

### 4.2 Variables a barrer
- Tamaño de matriz (n).
- Número de procesos/threads (p): potencias de 2 son cómodas para el análisis (1, 2, 4, 8, 16...), hasta el máximo de cores físicos disponibles. Corran también p=1 porque lo van a necesitar como referencia (ver sección 5).
- Estrategia de distribución de carga (row-wise vs bloques, si implementan ambas).
- Sistema de ejecución: PC personal vs servidor del laboratorio/cátedra (documenten specs: cantidad de cores, RAM, velocidad de clock, así explican diferencias entre sistemas).

### 4.3 Repeticiones y estadística
- Corran cada combinación (n, p, estrategia, sistema) **al menos 10 veces** (cuantas más, mejor, si el tiempo de cómputo lo permite).
- Descarten la primera corrida si sospechan efecto de "warm-up" de caché o de carga del sistema.
- Reporten: mediana, desvío estándar (o rango intercuartil) de los tiempos, no solo el promedio (el promedio es sensible a outliers por interferencia del SO).

---

## 5. Métricas a calcular (según lo pedido explícitamente)

Sea:
- `T_seq` = tiempo del algoritmo secuencial (mediana de varias corridas).
- `T_par(p)` = tiempo del algoritmo paralelo con p procesos (mediana de varias corridas).

**a) Speedup absoluto respecto de la solución secuencial (lineal):**
```
S_abs(p) = T_seq / T_par(p)
```
Este es el speedup "real", el que compara contra la mejor implementación no paralela.

**b) Speedup relativo respecto de la solución paralela con np = 1:**
```
S_rel(p) = T_par(1) / T_par(p)
```
Este mide qué tan bien escala tu implementación paralela en sí misma, aislando el overhead que introduce el propio framework paralelo (MPI/OpenMP) incluso corriendo con un solo proceso. Es normal (y esperable) que `T_par(1)` sea un poco mayor que `T_seq`, porque el código paralelo tiene overhead de inicialización aunque use un solo proceso — por eso este número interesa por separado.

**c) Desvío del speedup:**
Ya que calculan speedup con la mediana, también reporten la dispersión: pueden calcular el desvío estándar de los tiempos y propagarlo, o directamente calcular el speedup para cada corrida individual y reportar la mediana y el desvío del speedup mismo (más robusto).

**d) Eficiencia paralela:**
```
E(p) = S_abs(p) / p        (o E(p) = S_rel(p) / p, aclarar cuál usan y por qué)
```
Una eficiencia cercana a 1 indica escalado casi ideal; valores decrecientes con p indican que el overhead de comunicación/sincronización empieza a dominar.

**Sugerencia de tabla de resultados por cada tamaño n:**

| p | T_par (mediana) | Desvío | S_abs | S_rel | Eficiencia |
|---|---|---|---|---|---|
| 1 | ... | ... | ... | 1.00 | ... |
| 2 | ... | ... | ... | ... | ... |
| 4 | ... | ... | ... | ... | ... |
| ... | | | | | |

Y gráficos sugeridos:
- Curvas de speedup (S_abs y S_rel) vs p, una curva por tamaño de n, con la línea de speedup ideal (y=p) de referencia.
- Curva de eficiencia vs p.
- Tiempo de ejecución vs n para distintos p (escala log-log suele ayudar a ver el O(n³)).
- Si compararon estrategias de distribución: gráfico comparativo de speedup por estrategia, a igual n y p.

---

## 6. Análisis esperado / marco teórico para la discusión

- **Ley de Amdahl:** relacionen la caída de eficiencia con la fracción no paralelizable del código (inicialización, I/O, comunicación). Aunque el cómputo de matrices es "embarazosamente paralelo" en el core, la comunicación (broadcast/scatter/gather) es la parte que actúa como fracción serial/de overhead.
- **Por qué la eficiencia cae con p grande:** el trabajo por proceso (n/p filas) se reduce, pero el costo de comunicación (tamaño de B completa en cada proceso, en la versión row-wise) no baja proporcionalmente → a partir de cierto p, el overhead de comunicación domina sobre la ganancia de repartir cómputo.
- **Por qué n grande favorece el speedup:** a mayor n, el cómputo (O(n³)) crece más rápido que la comunicación (O(n²) para enviar una matriz), por lo que la relación cómputo/comunicación mejora y el speedup se acerca más al ideal.
- Comparen esto empíricamente con sus propios datos: ¿se ve más eficiencia en n=2048 que en n=256 para el mismo p? Esa observación es justamente la evidencia de lo anterior.

### 6.1 Referencias de biblioteca (reemplazan a la implementación 2D propia)

- **2D:** `PDGEMM` de ScaLAPACK (usa distribución block-cyclic 2D y un esquema tipo SUMMA/PUMMA; Choi et al. 1994, van de Geijn & Watts 1997), corrida sobre grilla Pr x Pc. Alternativas si se consiguen: COSMA, Cyclops (CTF), Elemental.
- **1D optimizada:** `PDGEMM` con grilla degenerada P x 1 (o 1 x P) equivale a una descomposición 1D por bloques de filas (o columnas); es la contraparte de biblioteca de nuestras versiones ijk/ikj/bloqueado MPI.
- **Cota superior local:** `dgemm` de BLAS optimizado (OpenBLAS/MKL) en np = 1 y como kernel local dentro de nuestra versión 1D.
- A definir/verificar: instalación disponible (WSL/servidor), enlazado, y que los tiempos se midan con el mismo esquema (semilla, repeticiones, descarte de la repetición 0). Citar en la bibliografía las fuentes efectivamente usadas.

---

## 7. Conclusiones (guía de qué deberían poder afirmar)

Al cerrar el informe, traten de responder con datos propios:
- Cuál fue el mejor speedup absoluto logrado, con qué (n, p, estrategia).
- A partir de qué p la eficiencia empieza a degradarse notablemente, y por qué (relacionarlo con comunicación/balance de carga).
- Si compararon estrategias de distribución, cuál ganó y en qué régimen (n chico vs n grande).
- Si compararon sistemas (PC vs servidor), qué diferencias de escalado observaron y a qué las atribuyen (cores físicos vs lógicos, velocidad de red/memoria, contención con otros procesos).

## 8. Mejoras recomendadas para trabajo futuro (para la pregunta final del enunciado)

Algunas ideas típicas y válidas para esta sección:
- Implementación propia de la distribución 2D (Cannon/SUMMA): queda fuera del alcance por decisión del equipo (no la desarrollamos ni podemos explicarla en detalle); se la reemplaza por una referencia de biblioteca (ver 6.1).
- Agregar tiling/blocking dentro de cada proceso para mejorar el uso de caché local.
- Usar comunicación no bloqueante (`MPI_Isend`/`MPI_Irecv`) para solapar comunicación con cómputo.
- Probar un esquema híbrido MPI+OpenMP.
- Usar bibliotecas optimizadas (BLAS/`dgemm`) como referencia de "cota superior" de rendimiento posible, comparando contra su propia implementación.
- Evaluar el uso de GPU (CUDA/OpenCL) como extensión natural del problema.
- Balanceo de carga dinámico si se usan matrices con estructura irregular (dispersas).

---

## 9. Bibliografía sugerida (punto de partida, agreguen lo que usen realmente)
- Blackford et al. — *ScaLAPACK Users' Guide* (SIAM, 1997), sobre PDGEMM y la distribución block-cyclic.
- Choi, Dongarra, Ostrouchov, Petitet, Walker, Whaley — "A proposal for a set of parallel basic linear algebra subprograms" / PUMMA.
- Grama, Gupta, Karypis, Kumar — *Introduction to Parallel Computing* (capítulo de multiplicación de matrices, es la referencia clásica de Cannon/SUMMA).
- Documentación oficial de MPI (mpi-forum.org) y de OpenMP (openmp.org).
- Documentación de la librería MPI que usen (Open MPI o MPICH) para las funciones colectivas (`MPI_Scatterv`, `MPI_Gatherv`, `MPI_Bcast`).
- Papers originales de Cannon (1969) y SUMMA (van de Geijn & Watts, 1997) si quieren profundizar en la parte 2D (solo como marco teórico de la referencia de biblioteca).

---

## 10. Checklist antes de la evaluación oral
- [ ] Código subido al repositorio (secuencial + paralelo + scripts de prueba/generación de datos).
- [ ] Datos crudos de todas las corridas (no solo el resumen) disponibles en el repo.
- [ ] Tablas y gráficos de speedup absoluto, speedup relativo y eficiencia.
- [ ] Al menos 2 sistemas distintos probados y comparados.
- [ ] Al menos 2 estrategias de distribución de carga comparadas (o justificar por qué solo una).
- [ ] Conclusiones redactadas con referencia directa a los datos obtenidos.
- [ ] Sección de mejoras futuras.
- [ ] Bibliografía citada.

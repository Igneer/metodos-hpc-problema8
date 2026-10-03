# Distribución 2D, SUMMA y OpenBLAS: explicación y ejemplos

La implementación principal continuará en C. OpenBLAS será un backend opcional e intercambiable del cálculo local: la distribución y SUMMA no dependerán de qué kernel realiza los productos.

Este documento explica los límites de cada componente y recorre una multiplicación completa con matrices concretas. No contiene código de implementación ni utiliza notación LaTeX.

El plan de implementación se encuentra en [plan_summa_2d.md](plan_summa_2d.md).

## 1. Tres responsabilidades diferentes

### Distribución 2D: quién posee los datos

Define cómo dividir las matrices y qué bloque guarda cada proceso.

Por ejemplo: el proceso situado arriba a la derecha posee el bloque superior derecho de A, de B y de C.

Eso todavía no explica cómo calcula C. Solo define propiedad y ubicación de datos. Hay que distinguir la descripción de esa propiedad de la transferencia que coloca los datos en cada proceso.

### SUMMA: cómo obtener los datos que faltan

Un proceso necesita bloques que inicialmente pertenecen a otros procesos. SUMMA establece:

1. Qué panel de A se comparte por cada fila de procesos.
2. Qué panel de B se comparte por cada columna.
3. Qué producto local se realiza.
4. Cómo se acumulan los productos hasta completar C.

**Distribuir en 2D no implica implementar SUMMA.** La misma distribución inicial puede utilizarse con otros algoritmos de comunicación.

### OpenBLAS: cómo multiplicar los operandos disponibles

Cuando un proceso tiene los operandos de una etapa, necesita ejecutar:

```text
C_local += panel_A × panel_B
```

Eso puede resolverlo un kernel ikj propio, un kernel bloqueado o OpenBLAS. OpenBLAS no decide dónde están los procesos ni distribuye paneles.

```text
Distribución 2D
    Define propiedad de bloques.

SUMMA
    Coordina broadcasts y acumulaciones.

Kernel local
    Ejecuta C_local += A_disponible × B_disponible.
    Puede ser propio u OpenBLAS.
```

## 2. Matrices del ejemplo

Se multiplicarán las siguientes matrices:

```text
A =

  1   2   3   4
  5   6   7   8
  9  10  11  12
 13  14  15  16
```

```text
B =

 1  0  2  0
 0  1  0  2
 3  0  4  0
 0  3  0  4
```

Se utilizarán cuatro procesos, organizados en una grilla de 2×2. Cada proceso tendrá bloques de 2×2.

Los procesos se nombrarán por su posición, no por su rank mundial:

```text
                Columna 0     Columna 1

Fila 0           P(0,0)        P(0,1)

Fila 1           P(1,0)        P(1,1)
```

Estos nombres son coordenadas lógicas. No suponen qué rank mundial corresponde a cada proceso.

## 3. Topología: organizar los procesos

Esta etapa prepara la infraestructura. Todavía no distribuye matrices ni multiplica.

### MPI_Dims_create: proponer la forma de la grilla

Recibe cuatro procesos y dos dimensiones. Para este caso propone dos filas y dos columnas de procesos.

Abstrae la elección de factores para organizar procesos. No calcula tamaños de bloques de matrices ni garantiza que la grilla elegida sea la mejor para cualquier forma de matriz.

### MPI_Cart_create: construir la grilla lógica

Crea un comunicador con estructura cartesiana. Para SUMMA se utilizará una grilla sin periodicidad: no se necesita conectar los extremos.

Abstrae la representación MPI de la organización 2D. No mueve A ni B ni crea una red física con esa forma.

Puede permitir reordenar ranks. Por eso no se deducirán coordenadas mediante divisiones o restos.

### MPI_Cart_coords: consultar las coordenadas

Cada proceso pregunta a MPI su posición dentro de la grilla:

```text
Un proceso obtiene (0,0).
Otro obtiene (0,1).
Otro obtiene (1,0).
Otro obtiene (1,1).
```

Abstrae la correspondencia entre rank cartesiano y coordenadas.

### MPI_Cart_sub: construir grupos de comunicación

Si el eje 0 representa filas y el eje 1 representa columnas:

- Conservar solo el eje de columnas crea comunicadores por fila.
- Conservar solo el eje de filas crea comunicadores por columna.

```text
Comunicadores de filas:

Fila 0: P(0,0), P(0,1)
Fila 1: P(1,0), P(1,1)
```

```text
Comunicadores de columnas:

Columna 0: P(0,0), P(1,0)
Columna 1: P(0,1), P(1,1)
```

En SUMMA, A se comparte dentro de las filas y B dentro de las columnas.

### MPI_Cart_rank: identificar al propietario

Cuando se necesita al proceso de una coordenada, MPI proporciona su rank. También se consulta esta correspondencia dentro de los subcomunicadores para seleccionar el root de cada broadcast.

No se asumirá que columna 1 significa rank 1 en cualquier comunicador. Un rank siempre pertenece al contexto de un comunicador concreto.

### Por qué SUMMA no necesita MPI_Cart_shift

MPI_Cart_shift proporciona origen y destino para un desplazamiento cartesiano; no mueve datos. El SUMMA de este documento utiliza broadcasts, no intercambios entre vecinos, por lo que no necesita esa función.

Si una operación futura requiere vecinos, se obtendrán con MPI_Cart_shift, nunca con fórmulas manuales de ranks.

## 4. Partición: definir la propiedad de los bloques

La matriz A se divide así:

```text
A00 =                 A01 =

 1  2                  3  4
 5  6                  7  8


A10 =                 A11 =

  9  10               11  12
 13  14               15  16
```

La matriz B se divide así:

```text
B00 =                 B01 =

 1  0                  2  0
 0  1                  0  2


B10 =                 B11 =

 3  0                  4  0
 0  3                  0  4
```

| Proceso | A inicial | B inicial | Resultado a completar |
|---------|-----------|-----------|------------------------|
| P(0,0) | A00 | B00 | C00 |
| P(0,1) | A01 | B01 | C01 |
| P(1,0) | A10 | B10 | C10 |
| P(1,1) | A11 | B11 | C11 |

Tener esos bloques no alcanza para multiplicarlos simplemente entre sí:

```text
C00 = A00 × B00 + A01 × B10
C01 = A00 × B01 + A01 × B11
C10 = A10 × B00 + A11 × B10
C11 = A10 × B01 + A11 × B11
```

P(0,0), por ejemplo, tiene A00 y B00, pero le faltan A01 y B10. Aquí aparece la necesidad de SUMMA.

Se reutilizará calcular_particiones() para describir cortes balanceados de los ejes. MPI no sustituye toda la planificación de bloques y segmentos: esa lógica pertenece al algoritmo, no al cálculo de vecinos.

## 5. Transferencia inicial: materializar la distribución

La matriz A completa se almacena por filas:

```text
1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
```

A01 contiene:

```text
3 4
7 8
```

Esos cuatro valores no forman un tramo contiguo en A global. Entre las filas del bloque están otros elementos.

### MPI_Type_create_subarray: describir el rectángulo

Para A01 se indica:

```text
Matriz global:       4×4
Tamaño del bloque:   2×2
Inicio:             fila 0, columna 2
Orden de memoria:   por filas, MPI_ORDER_C
Tipo de elemento:   MPI_DOUBLE
```

MPI identifica los elementos 3, 4, 7 y 8. El receptor puede recibirlos como cuatro doubles contiguos:

```text
A_local de P(0,1):

3 4 7 8
```

El datatype abstrae el recorrido de una región no contigua durante la transferencia. No garantiza ausencia de copias internas: MPI puede empaquetar esos datos.

Se utilizarán:

- MPI_Type_create_subarray para describir el bloque.
- MPI_Type_commit para preparar el datatype.
- MPI_Send para transmitir la región descrita.
- MPI_Recv para recibir el bloque local contiguo.

El bloque del propio root puede transferirse localmente mediante MPI_Sendrecv con MPI_COMM_SELF, encapsulado en el módulo de transferencias.

Al terminar, cada proceso tiene los bloques iniciales de la tabla. **Ya existe una distribución 2D, pero todavía no se ha ejecutado SUMMA.**

## 6. Preparar el resultado

Todos inicializan su bloque de C:

```text
C_local =

0 0
0 0
```

Se reutilizará matriz_poner_cero(). La inicialización ocurre una vez por multiplicación completa, no antes de cada aporte, porque SUMMA acumula varios productos.

## 7. Primera etapa de SUMMA: índice de bloques 0

Este ejemplo tiene dos etapas uniformes. La primera utiliza el primer tramo de la dimensión compartida.

### Compartir A por filas

En la fila 0, P(0,0) difunde A00:

```text
P(0,0) y P(0,1) disponen de:

1 2
5 6
```

En la fila 1, P(1,0) difunde A10:

```text
P(1,0) y P(1,1) disponen de:

 9 10
13 14
```

Se utiliza MPI_Bcast sobre cada comunicador de fila.

### Compartir B por columnas

En la columna 0, P(0,0) difunde B00:

```text
P(0,0) y P(1,0) disponen de:

1 0
0 1
```

En la columna 1, P(0,1) difunde B01:

```text
P(0,1) y P(1,1) disponen de:

2 0
0 2
```

Se utiliza MPI_Bcast sobre cada comunicador de columna.

### Qué hace MPI_Bcast

Un propietario aporta datos y todos los miembros del comunicador obtienen ese contenido. No multiplica ni necesita una lógica de todos enviándose datos entre sí.

SUMMA decide qué contenido difundir y en qué grupo. MPI ejecuta la comunicación. Todos los miembros del grupo deben participar en un orden compatible; un broadcast no sustituye conceptualmente a una barrera.

### Multiplicar localmente

| Proceso | Producto de la primera etapa |
|---------|------------------------------|
| P(0,0) | A00 × B00 |
| P(0,1) | A00 × B01 |
| P(1,0) | A10 × B00 |
| P(1,1) | A10 × B01 |

B00 es la identidad y B01 es dos veces la identidad. Los acumuladores quedan:

```text
C00 =                 C01 =

1 2                   2  4
5 6                  10 12


C10 =                 C11 =

 9 10                 18 20
13 14                 26 28
```

Estos productos son responsabilidad del kernel local, no de MPI.

## 8. Segunda etapa de SUMMA: índice de bloques 1

### Compartir A por filas

Los propietarios son ahora los procesos de la columna 1:

```text
Fila 0: P(0,1) difunde A01

3 4
7 8


Fila 1: P(1,1) difunde A11

11 12
15 16
```

### Compartir B por columnas

Los propietarios son ahora los procesos de la fila 1:

```text
Columna 0: P(1,0) difunde B10

3 0
0 3


Columna 1: P(1,1) difunde B11

4 0
0 4
```

Se utiliza MPI_Bcast, cambiando los roots mediante consultas cartesianas.

### Acumular el segundo aporte

| Proceso | Operación |
|---------|-----------|
| P(0,0) | C00 += A01 × B10 |
| P(0,1) | C01 += A01 × B11 |
| P(1,0) | C10 += A11 × B10 |
| P(1,1) | C11 += A11 × B11 |

Para P(0,0):

```text
Acumulado anterior:

1 2
5 6

Nuevo aporte:

 9 12
21 24

Resultado:

10 14
26 30
```

Los bloques finales son:

```text
C00 =                 C01 =

10 14                 14 20
26 30                 38 44


C10 =                 C11 =

42 46                 62 68
58 62                 86 92
```

Cada proceso completó su resultado. No se necesita sumar bloques de procesos distintos para completar C.

## 9. Recolección: reconstruir C completa

Cada proceso envía su bloque contiguo. El root recibe mediante un datatype de subarray que describe su posición global:

```text
C00 → filas 0–1, columnas 0–1
C01 → filas 0–1, columnas 2–3
C10 → filas 2–3, columnas 0–1
C11 → filas 2–3, columnas 2–3
```

El resultado es:

```text
C =

10 14 14 20
26 30 38 44
42 46 62 68
58 62 86 92
```

Un elemento puede verificarse directamente:

```text
C[0,0] = 1×1 + 2×0 + 3×3 + 4×0
       = 10
```

Se utilizan MPI_Send desde los bloques locales y MPI_Recv con los datatypes de las regiones globales.

No se utiliza MPI_Reduce para construir C: los bloques son regiones diferentes, no contribuciones superpuestas. MPI_Reduce con MPI_MAX sí sirve para obtener el mayor tiempo local de ejecución.

## 10. OpenBLAS como backend intercambiable

La interfaz del kernel local recibirá:

- Operandos A y B disponibles.
- Acumulador C.
- Cantidad de filas.
- Ancho de la dimensión compartida.
- Cantidad de columnas.
- Strides de A, B y C, expresados en elementos.

Su contrato será:

```text
C += A × B
```

Las implementaciones podrán ser un kernel propio ikj, uno bloqueado o un adaptador OpenBLAS. SUMMA invocará esa operación sin conocer la implementación. En C puede seleccionarse mediante un puntero a función proporcionado por el driver.

### Aplicación al ejemplo

Cada producto local tiene dos filas, dimensión compartida de ancho dos y dos columnas.

El adaptador OpenBLAS utilizará cblas_dgemm con orden por filas, sin transponer los operandos, alpha = 1 y beta = 1.

GEMM realiza:

```text
C = alpha × A × B + beta × C
```

Con esos valores:

```text
C = A × B + C
```

**beta = 1 conserva los aportes anteriores.** Si se utilizara beta = 0 en todas las etapas, la segunda borraría la primera. Como C se inicializa en cero, el mismo contrato sirve para todas las etapas.

### Por qué la interfaz admite strides

En el ejemplo principal los paneles coinciden con bloques completos. En el caso general puede requerirse solo una parte del bloque local:

```text
Bloque local:

1 2 3 4
5 6 7 8

Panel requerido:

3 4
7 8
```

En el propietario, el panel tiene ancho dos, pero los comienzos de sus filas están separados por cuatro elementos. En un receptor con panel compacto, están separados por dos.

MPI puede transmitir el panel mediante un datatype derivado. El propietario puede conservar una vista sobre su bloque original y los receptores pueden utilizar paneles contiguos. Las firmas de tipos comunicadas deben coincidir aunque los layouts sean diferentes.

OpenBLAS puede consumir ambos layouts mediante leading dimensions correctos. No debe suponerse que el stride de A siempre coincide con el ancho del panel.

### Separación de dependencias

```text
Driver
 ├── Partición 2D
 ├── Topología MPI
 ├── Transferencias MPI
 └── SUMMA
      └── Interfaz de kernel
           ├── Kernel propio
           └── Adaptador OpenBLAS
```

Solo el adaptador OpenBLAS necesita conocer cblas.h y enlazar esa biblioteca. Los demás módulos pueden compilar y probarse sin ella.

## 11. Generalización a grillas rectangulares

El proyecto admite A de N×M y B de M×P, con dimensiones fijadas al compilar. El valor por defecto cuadrado no limita el contrato general.

Con dimensión compartida de tamaño seis y una grilla configurada de 2×3:

```text
Columnas de A:
[0,2), [2,4), [4,6)

Filas de B:
[0,3), [3,6)
```

La notación [inicio, fin) incluye inicio y excluye fin. Los cortes no coinciden, por lo que se fusionan:

```text
Segmentos comunes:
[0,2), [2,3), [3,4), [4,6)

Anchos de las cuatro etapas:
2, 1, 1, 2
```

Cada etapa utiliza exactamente las mismas posiciones de la dimensión compartida en A y B. La nueva implementación conservará la lógica de fusión del SUMMA existente.

MPI resolverá los ranks propietarios, describirá paneles y realizará broadcasts. La fusión de cortes seguirá siendo planificación del algoritmo.

## 12. Límites de los módulos

| Módulo | Responsabilidad | No le corresponde |
|--------|-----------------|-------------------|
| Partición 2D | Tamaños, offsets y propiedad lógica | Calcular vecinos por fórmulas |
| Topología | Coordenadas, ranks y grupos mediante MPI | Multiplicar o distribuir valores |
| Transferencias | Datatypes y movimiento de bloques | Decidir la secuencia SUMMA |
| SUMMA | Orden de paneles, broadcasts y acumulaciones | Conocer detalles de OpenBLAS |
| Kernel local | Producto y acumulación con strides | Comunicarse con otros procesos |
| Driver | Entrada, buffers, backend, validación y tiempos | Duplicar la lógica de los módulos |

Los datatypes creados se liberan con MPI_Type_free y los comunicadores propios con MPI_Comm_free, antes de MPI_Finalize. Los buffers estáticos de la aplicación no requieren free.

OpenBLAS puede administrar memoria interna propia: conservar matrices estáticas de aplicación no significa prohibir toda asignación dinámica dentro de las bibliotecas.

Para comparar distribuciones se debe utilizar el mismo backend local cuando sea posible, y estudiar los cambios de kernel por separado. Se comenzará con un hilo de OpenBLAS por proceso MPI, controlado según el tipo de build de la biblioteca: OPENBLAS_NUM_THREADS en builds habituales y OMP_NUM_THREADS en builds OpenMP.

**Idea central:** primero se establece dónde viven los datos; después SUMMA organiza cómo se comparten; finalmente el kernel propio u OpenBLAS realiza las cuentas locales.

## Referencias

- [Open MPI: funciones MPI](https://docs.open-mpi.org/en/main/man-openmpi/man3/index.html).
- [Open MPI: MPI_Type_create_subarray](https://docs.open-mpi.org/en/main/man-openmpi/man3/MPI_Type_create_subarray.3.html).
- [Open MPI: MPI_Bcast](https://docs.open-mpi.org/en/main/man-openmpi/man3/MPI_Bcast.3.html).
- [OpenBLAS: manual de usuario](https://github.com/OpenMathLib/OpenBLAS/blob/develop/docs/user_manual.md).
- [OpenBLAS: control de hilos](https://github.com/OpenMathLib/OpenBLAS/blob/develop/README.md).

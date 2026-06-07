# Comms-System

## TP2: Modulacion y demodulacion

El TP2 trabaja sobre el vector binario que se obtiene al codificar el texto con Huffman en el TP1. En `main.py`, esa salida se convierte en un arreglo de bits:

```python
binary_vector = np.array([int(b) for b in ''.join(codified_text)])
```

Luego ese vector entra a la funcion `tp2`, donde se define el tipo de modulacion, la cantidad de simbolos de la constelacion y el tipo de etiquetado:

```python
modulation_type, M, code_label = "QAM", 16, "Binary"
```

En esta configuracion se usa una modulacion `16-QAM`, por lo que cada simbolo transporta:

```python
k = log2(M) = log2(16) = 4 bits
```

### Transmisor

La parte del transmisor esta implementada en `scripts/transmitter.py`, principalmente en la funcion `modulate_symbols`.

Esta funcion recibe:

- `binary_vector`: vector de bits a transmitir.
- `modulation_type`: tipo de modulacion, puede ser `"QAM"` o `"FSK"`.
- `M`: cantidad de simbolos posibles.
- `code_label`: tipo de etiquetado, puede ser `"Binary"` o `"Gray"`.

Primero se valida que los parametros sean correctos. El valor de `M` debe ser una potencia de 2 entre 2 y 16. Despues se calcula la cantidad de bits por simbolo:

```python
k = int(np.log2(M))
```

Si la longitud del vector binario no es multiplo de `k`, se agregan ceros al final para poder agrupar los bits en palabras completas de `k` bits. Luego cada grupo se convierte a decimal para obtener el indice del simbolo que se va a transmitir.

#### Modulacion QAM

Si `modulation_type == "QAM"`, cada simbolo se representa como un punto en el plano con dos componentes:

- componente en fase, o eje `I`;
- componente en cuadratura, o eje `Q`.

El codigo divide los `k` bits del simbolo entre ambos ejes. Para cada posible simbolo de la constelacion se calculan sus coordenadas usando `amplitude_label`, que transforma los bits de cada eje en un nivel de amplitud.

Despues se normaliza la constelacion para que la energia media por simbolo coincida con:

```python
Es = Eb * k
```

En el codigo se toma `Eb = 1`, por lo que para `16-QAM` la energia por simbolo es `Es = 4`.

Finalmente, el transmisor devuelve los puntos de la constelacion correspondientes a los simbolos del mensaje.

#### Modulacion FSK

Si `modulation_type == "FSK"`, cada simbolo se representa con un vector de `M` componentes. Todas las posiciones valen cero excepto la posicion correspondiente al indice del simbolo, que toma el valor:

```python
sqrt(Es)
```

Es decir, en FSK se transmite energia en una sola frecuencia posible por simbolo.

### Receptor

La parte del receptor esta implementada en `scripts/receiver.py`, principalmente en la funcion `demodulate_symbols`.

Esta funcion recibe la senal modulada y usa los mismos parametros que el transmisor:

- `received_signal`: senal recibida.
- `modulation_type`: tipo de modulacion usado.
- `M`: cantidad de simbolos posibles.
- `code_label`: tipo de etiquetado usado.
- `original_length`: longitud original del vector binario antes del padding.

El receptor vuelve a calcular:

```python
k = int(np.log2(M))
Es = Eb * k
```

con el mismo criterio usado por el transmisor. Esto es necesario para poder reconstruir la misma referencia de decision.

#### Demodulacion QAM

Si la modulacion es QAM, el receptor reconstruye dentro de `demodulate_symbols` la misma constelacion que uso el transmisor. Para eso repite el calculo de los puntos posibles en los ejes `I` y `Q`, usando el mismo etiquetado binario o Gray.

Luego compara cada punto recibido contra todos los puntos posibles de la constelacion. La decision se hace por distancia minima:

```python
symbols_idx = np.argmin(distances, axis=1)
```

Esto significa que cada punto recibido se interpreta como el simbolo de la constelacion mas cercano.

#### Demodulacion FSK

Si la modulacion es FSK, el receptor busca la componente de mayor valor en cada simbolo recibido:

```python
symbols_idx = np.argmax(received_signal, axis=1)
```

Ese indice indica cual fue la frecuencia activa y, por lo tanto, que simbolo fue transmitido.

### Recuperacion de bits

Una vez obtenidos los indices de los simbolos, el receptor convierte cada indice decimal nuevamente a una palabra binaria de `k` bits:

```python
binary_vector = np.concatenate([decimal2binary(symbol_idx, k) for symbol_idx in symbols_idx])
```

Si durante la modulacion se agregaron ceros de padding, se eliminan usando `original_length`:

```python
binary_vector = binary_vector[:original_length]
```

De esta forma, la salida de `demodulate_symbols` tiene la misma longitud que el vector binario original.

### Verificacion

En `main.py`, despues de modular y demodular, se compara el vector original contra el vector recuperado:

```python
np.array_equal(binary_vector, demodulated_binary_vector)
```

Si ambos son iguales, el programa imprime:

```text
Demodulation successful: True
```

Esto confirma que, sin ruido en el canal, el receptor recupera correctamente los bits enviados por el transmisor.

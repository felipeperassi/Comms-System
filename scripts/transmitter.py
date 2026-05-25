import numpy as np

def appearence_probs(data, display_counts=False, display_probas=False): # we could use counter instead 
  """
  Calculates the appearance probability of each character in a string.

    Parameters:
        data: str, the text to be processed
        display_counts: bool, if True prints the character counts
        display_probas: bool, if True prints the character probabilities

    Returns:
        dict {character: probability}
  """

  char_counts_dict = { }
  for char in data:
      char_counts_dict[char] = char_counts_dict.get(char, 0) + 1
    
  # Optionally, sort the counts for better readability
  sorted_char_counts = dict(sorted(char_counts_dict.items()))

  if display_counts:
    print("Character counts:")
    for char, count in sorted_char_counts.items():
        # Handle newline and tab characters for clearer printing
        if char == '\n':
            print(f"'\\n' (newline): {count}")
        elif char == '\t':
            print(f"'\\t' (tab): {count}")
        else:
            print(f"'{char}': {count}")

  probs_dict = {}
  total_chars = len(data)
  for char, count in sorted_char_counts.items():
      probs_dict[char] = count / total_chars

  if display_probas:
    print("Character probs:")
    for char, prob in probs_dict.items():
        if char == '\n':
            print(f"'\\n' (newline): {prob:.4f}")
        elif char == '\t':
            print(f"'\\t' (tab): {prob:.4f}")
        else:
            print(f"'{char}': {prob:.4f}")

  return probs_dict

def entropy(probs_dict):
    """
    Calculates the entropy of a source given its probability distribution.

    Parameters:
        probs_dict: dict {symbol: probability}

    Returns:
        float: entropy in bits
    """
    return np.sum([-prob * np.log2(prob) for prob in probs_dict.values()])

def huffman_algorithm(probs_dict):
    """
    Calculates the Huffman binary code for each symbol in the given probability dictionary.

    Parameters:
        probs_dict: dict {symbol: probability}

    Returns:
        dict {symbol: binary code}
    """
    nodes = [[proba, symbol, None, None] for symbol, proba in probs_dict.items()]
    nodes.sort(key=lambda x: x[0])  # sort by proba

    while len(nodes) > 1:
        left = nodes.pop(0)  # smallest proba
        right = nodes.pop(0)  # 2nd smallest proba
        new_node = [left[0] + right[0], None, left, right]  
        nodes.append(new_node)  
        nodes.sort(key=lambda x: x[0])  # sort again (if draw, the older one goes first in pop)

    code_dict = {}
    stack = [(nodes[0], "")] # root node, empty prefix

    while stack: # LIFO stack for DFS
        node, prefix = stack.pop()
        if node[1] is not None: # sym != None -> leaf node  
            code_dict[node[1]] = prefix
        else: # sym = None -> internal node
            stack.append((node[3], prefix + "1"))
            stack.append((node[2], prefix + "0"))

    return code_dict

# testing:
# probs = {"a": 0.35, "b": 0.17, "c": 0.17, "d": 0.10, "e": 0.10, "f": 0.07, "g": 0.04}
# codes = huffman_algorithm(probs)
# for symbol, code in codes.items():
#     print(f"{symbol}: {code}")

def mean_length(code_dict, probs_dict):
    """
    Calculates the mean code length of a Huffman code.
    
    Parameters:
        code_dict: dict {symbol: code}
        probs_dict: dict {symbol: probability}
    
    Returns:
        float: expected number of bits per symbol
    """
    return sum(probs_dict[sym] * len(code) for sym, code in code_dict.items())

def minimum_length(code_dict):
    """
    Calculates the minimum code length of a Huffman code.
    
    Parameters:
        code_dict: dict {symbol: code}
    
    Returns:
        int: length of the longest code
    """
    return min(len(code) for code in code_dict.values())

def shannon_range(entropy_value, m=2):
    """
    Calculates the Shannon range for a given entropy value.
    
    Parameters:
        entropy_value: float, the entropy of the source
        m: int, the number of symbols in the alphabet (default is 2 for binary codes)
    
    Returns:
        tuple: (lower_bound, upper_bound)
    """
    return (entropy_value / np.log2(m), (entropy_value / np.log2(m)) + 1)

# ---- Earlier version of huffman_algorithm ----

# def huffman_algorithm2(probs):

#     counter = 0  # contador para desempatar nodos con igual probabilidad

#     # Crear un nodo por cada símbolo: [prob, contador_unico, simbolo, hijo_izq, hijo_der]
#     # Las hojas no tienen hijos, por eso None, None al final
#     heap = []
#     for sym, prob in probs.items():
#         counter += 1
#         heap.append([prob, counter, sym, None, None])

#     # Ordenar el heap: el de menor probabilidad queda primero
#     heapq.heapify(heap)

#     # Repetir hasta que quede un solo nodo (la raíz)
#     while len(heap) > 1:

#         # Sacar el nodo de menor probabilidad
#         left = heapq.heappop(heap)

#         # Sacar el siguiente nodo de menor probabilidad
#         right = heapq.heappop(heap)

#         # Crear un nodo padre que los combine
#         # su prob es la suma de los dos hijos
#         # su símbolo es None porque no representa ningún caracter
#         counter += 1
#         padre = [left[0] + right[0], counter, None, left, right]

#         # Insertar el padre de vuelta en el heap
#         heapq.heappush(heap, padre)

#     # Cuando queda uno solo, ese es la raíz del árbol
#     raiz = heap[0]

#     # Recorrer el árbol y asignar códigos binarios
#     codigos = {}
#     _asignar_codigos(raiz, '', codigos)
#     return codigos


# def _asignar_codigos(nodo, codigo_actual, codigos):

#     # Si el nodo tiene símbolo, es una hoja → guardar su código
#     if nodo[2] is not None:
#         codigos[nodo[2]] = codigo_actual
#         return

#     # Bajar al hijo izquierdo agregando '0' al código
#     _asignar_codigos(nodo[3], codigo_actual + '0', codigos)

#     # Bajar al hijo derecho agregando '1' al código
#     _asignar_codigos(nodo[4], codigo_actual + '1', codigos)
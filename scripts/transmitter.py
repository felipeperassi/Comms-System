import numpy as np

# ----------------------------------- TP1 -----------------------------------

def appearence_probs(text) -> tuple:
  """
  Calculates the appearance probability of each character in a string.

    Parameters:
        text: str, the text to be processed

    Returns:
        dict {character: probability}
        dict {character: count}
  """

  char_counts_dict = { }
  for char in text:
      char_counts_dict[char] = char_counts_dict.get(char, 0) + 1
    
  sorted_char_counts = dict(sorted(char_counts_dict.items())) # Optionally, sort the counts for better readability

  probs_dict = {}
  total_chars = len(text)
  for char, count in sorted_char_counts.items():
      probs_dict[char] = count / total_chars

  return probs_dict, char_counts_dict

def entropy(probs_dict) -> float:
    """
    Calculates the entropy of a source given its probability distribution.

    Parameters:
        probs_dict: dict {symbol: probability}

    Returns:
        float: entropy in binary_vector
    """
    return np.sum([-prob * np.log2(prob) for prob in probs_dict.values()])

def huffman_algorithm(probs_dict) -> dict:
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
            stack.append((node[3], prefix + "0"))
            stack.append((node[2], prefix + "1"))

    return code_dict

def mean_length(code_dict, probs_dict) -> float:
    """
    Calculates the mean code length of a Huffman code.
    
    Parameters:
        code_dict: dict {symbol: code}
        probs_dict: dict {symbol: probability}
    
    Returns:
        float: expected number of binary_vector per symbol
    """
    return sum(probs_dict[sym] * len(code) for sym, code in code_dict.items())

def minimum_length(code_dict) -> int:
    """
    Calculates the minimum code length of a Huffman code.
    
    Parameters:
        code_dict: dict {symbol: code}
    
    Returns:
        int: length of the longest code
    """
    return min(len(code) for code in code_dict.values())

def shannon_range(entropy_value, m=2) -> tuple:
    """
    Calculates the Shannon range for a given entropy value.
    
    Parameters:
        entropy_value: float, the entropy of the source
        m: int, the number of symbols in the alphabet (default is 2 for binary codes)
    
    Returns:
        tuple: (lower_bound, upper_bound)
    """
    return (entropy_value / np.log2(m), (entropy_value / np.log2(m)) + 1)

def codificate_text(text, code_dict) -> list:
    """
    Codifies a text using the provided code dictionary.
    
    Parameters:
        text: str, the text to be codified
        code_dict: dict {symbol: code}
    
    Returns:
        list: the codified binary codes for each character in the text
    """
    return [code_dict[char] for char in text]

# ----------------------------------- TP2 -----------------------------------

def gray2binary(binary_vector) -> np.array:
    """"
    Converts a vector of binary_vector in Gray code to binary code.

    Parameters:
        binary_vector: np.array, the input vector of binary_vector in Gray code
    Returns:
        np.array: the output vector of binary_vector in binary code
    """
    binary = np.zeros_like(binary_vector)
    binary[0] = binary_vector[0]
    for i in range(1, len(binary_vector)):
        binary[i] = binary[i-1] ^ binary_vector[i]

    return binary

def binary2decimal(binary_vector) -> int:
    """
    Converts a vector of binary numbers to a decimal number.

    Parameters:
        binary_vector: np.array, the input vector of binary numbers (e.g., [1, 0, 1] for the binary number "101")

    Returns:
        int: the decimal number
    """
    return int(''.join(map(str, binary_vector)), 2)

def amplitude_label(bits_axis, n_levels, code_label) -> float:
    """
    Converts a vector of binary_vector to an amplitude level based on the specified code label.

    Parameters:
        bits_axis: np.array, the input vector of binary_vector representing the symbol index
        n_levels: int, the number of amplitude levels (M)
        code_label: str, the type of code ("Gray" or "Binary")

    Returns:
        float: the amplitude level corresponding to the input binary_vector
    """
    if len(bits_axis) == 0:
        return 0

    if code_label == "Gray":
        bits_axis = gray2binary(bits_axis)

    pos = binary2decimal(bits_axis)
    return 2 * pos - (n_levels - 1)

def modulate_symbols(binary_vector, modulation_type, M, code_label) -> tuple:
    """
    Modulates a binary vector into a constellation based on the specified modulation type, M-ary level and code label.

    Parameters:
        binary_vector: np.array, the input vector with a binary representation
        modulation_type: str, the type of modulation ("QAM" or "FSK")
        M: int, the number of symbols in the constellation
        code_label: str, the type of code ("Gray" or "Binary")

    Returns:
        np.array: the modulated constellation points corresponding to the input characters
        np.array: the indices of the modulated symbols
    """
    supported_types = ["QAM", "FSK"]
    if modulation_type not in supported_types:
        raise ValueError(f"Invalid modulation type. Expected one of {supported_types}, got {modulation_type}")

    labels = ["GRAY", "BINARY"]
    if code_label.upper() not in labels:
        raise ValueError(f"Invalid code label. Expected one of {labels}, got {code_label}")
    if modulation_type == "FSK" and code_label.upper() == "GRAY":
        raise ValueError("Gray code is not applicable for FSK modulation.")

    if M < 2 or M > 16 or np.log2(M) % 1 != 0:
        raise ValueError("M must be a power of 2 between 2 and 16 inclusive.")

    Eb = 1                  # Energy per bit
    k = int(np.log2(M))     # Bits per symbol
    Es = Eb * k             # Energy per symbol

    """CHEQUEAR ESTO SI SE AGREGAN LOS CEROS AL PRINCIPIO O AL FINAL"""
    if len(binary_vector) % k != 0: # Vector length must be a multiple of k
        binary_vector = np.pad(binary_vector, (0, k - (len(binary_vector) % k)), mode='constant')

    symbols = binary_vector.reshape(-1, k) # (N words x k binary_vector)
    symbols_idx = np.array([binary2decimal(row) for row in symbols]) # decimal representation of the N words (N x 1)

    if modulation_type == "QAM":
        kI, kQ = int(np.ceil(k / 2)), int(np.floor(k / 2)) # Round up & down
        MI, MQ = 2 ** kI, 2 ** kQ

        constellation = np.zeros((M, 2))
        for number in range(M):
            binary_number = np.array(list(np.binary_repr(number, width=k)), dtype=int) # decimal to binary
            coordI = amplitude_label(binary_number[:kI], MI, code_label)
            coordQ = amplitude_label(binary_number[kI:], MQ, code_label)
            constellation[number] = [coordI, coordQ]

        Es_grid = np.mean(np.sum(constellation**2, axis=1))
        constellation *= np.sqrt(Es / Es_grid)

        return constellation[symbols_idx], symbols_idx # (N x 2)

    if modulation_type == "FSK":
        coords = np.zeros((len(symbols), M))
        coords[np.arange(len(symbols)), symbols_idx] = np.sqrt(Es)

        return coords, symbols_idx # (N x M)

def calculate_energies(modulated_signal, M) -> tuple:
    """
    Calculates the energy of a modulated signal.

    Parameters:
        modulated_signal: np.array, the input modulated signal (N x 2 for QAM or N x M for FSK)
        M: int, the number of symbols in the constellation

    Returns:
        tuple: (Es, Eb) where Es is the energy per symbol and Eb is the energy per bit
    """

    Es = np.sum(modulated_signal**2, axis=1) # Energy per symbol
    Eb = Es / np.log2(M) # Energy per bit

    return Es, Eb

def calculate_mean_energies(modulated_signal, M) -> tuple:
    """
    Calculates the mean energies of a modulated signal.

    Parameters:
        modulated_signal: np.array, the input modulated signal (N x 2 for QAM or N x M for FSK)
        M: int, the number of symbols in the constellation

    Returns:
        tuple: (Es_mean, Eb_mean) where Es_mean is the mean energy per symbol and Eb_mean is the mean energy per bit
    """

    Es, Eb = calculate_energies(modulated_signal, M)

    return np.mean(Es), np.mean(Eb)

# ----------------------------------- TP4 -----------------------------------

def encode_block(message, G) -> np.array:
    """
    Encodes a block of k bits into a codeword of n bits using the generator matrix G.
    
    Parameters:
        message: np.array, the input block of k bits (shape (k,))
        G: np.array, the generator matrix of shape (k, n)

    Returns:
        np.array: the encoded codeword of n bits (shape (n,))
    """

    return np.mod(message @ G, 2)


def codificate_channel(binary_vector, G, k, n) -> np.array:
    """
    Codifies a binary vector using the channel coding scheme defined by the generator matrix G.
    
    Parameters:
        binary_vector: np.array, the input binary vector to be codified
        G: np.array, the generator matrix of shape (k, n)
        k: int, the number of bits in each message block
        n: int, the number of bits in each codeword
    
    Returns:
        np.array: the codified binary vector
    """

    if len(binary_vector) % k != 0: # Vector length must be a multiple of k
        binary_vector = np.concatenate([binary_vector, np.zeros(k - len(binary_vector) % k, dtype=int)])

    blocks = binary_vector.reshape(-1, k) # k bits blocks (N x k)
    encoded_blocks = np.array([encode_block(block, G) for block in blocks]) # Encode each block (N x n)
    return encoded_blocks.flatten()


if __name__ == "__main__":
    # Genera los diagramas de constelacion QAM transmitidos (sin ruido) para
    # M = 2, 4, 8, 16, reutilizando el mismo flujo de codificacion y modulacion.
    from data.config import TXT_PATH, MEDIA_PATH
    from scripts.extras import plot_constellation

    M_VEC = [2, 4, 8, 16]        # niveles QAM a graficar
    CODE_LABEL = "Binary"        # mapeo de bits ("Binary" o "Gray")
    OUT_DIR = MEDIA_PATH / "constellations"

    with open(TXT_PATH, 'r') as f:
        text = f.read()

    # Codificacion de fuente (mismo flujo que el modem)
    probs_dict, _ = appearence_probs(text)
    code_dict = huffman_algorithm(probs_dict)
    codified_text = codificate_text(text, code_dict)
    binary_vector = np.array([int(bit) for symbol in codified_text for bit in symbol])

    energy_rows = []   # (M, k, Es_teo, Es_hat, Eb_teo, Eb_hat)
    for M in M_VEC:
        mod_symbols, _ = modulate_symbols(binary_vector, "QAM", M, CODE_LABEL)
        plot_constellation("QAM", mod_symbols, M, OUT_DIR,
                           code_label=CODE_LABEL, filename=f"qam_M{M}_transmitted.png")
        print(f"QAM M={M}: constelacion transmitida -> {OUT_DIR / f'qam_M{M}_transmitted.png'}")

        # Energias: teoricas (Eb=1 -> Es=k) vs estimadas sobre los simbolos enviados
        k = int(np.log2(M))
        Eb_teo, Es_teo = 1.0, float(k)
        Es_hat, Eb_hat = calculate_mean_energies(mod_symbols, M)
        energy_rows.append((M, k, Es_teo, Es_hat, Eb_teo, Eb_hat))

    # Energias FSK (M = 2, 4, 8, 16). No se grafican: para M > 2 la constelacion
    # ortogonal vive en M dimensiones y no es representable en el plano.
    fsk_rows = []
    for M in M_VEC:
        mod_symbols, _ = modulate_symbols(binary_vector, "FSK", M, "Binary")
        k = int(np.log2(M))
        Eb_teo, Es_teo = 1.0, float(k)
        Es_hat, Eb_hat = calculate_mean_energies(mod_symbols, M)
        fsk_rows.append((M, k, Es_teo, Es_hat, Eb_teo, Eb_hat))

    def print_energy_table(title, rows):
        print(f"\nEnergia media por simbolo y por bit (teorica vs estimada) - {title}")
        print(f"{'M':>3} {'k':>3} {'Es_teo':>8} {'Es_hat':>8} {'Eb_teo':>8} {'Eb_hat':>8}")
        print("-" * 42)
        for M, k, Es_teo, Es_hat, Eb_teo, Eb_hat in rows:
            print(f"{M:>3} {k:>3} {Es_teo:>8.4f} {Es_hat:>8.4f} {Eb_teo:>8.4f} {Eb_hat:>8.4f}")

    # Tablas comparativas de energias medias por simbolo y por bit
    print_energy_table("QAM", energy_rows)
    print_energy_table("FSK", fsk_rows)

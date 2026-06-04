import numpy as np

# TP1
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
        float: entropy in bits
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
            stack.append((node[3], prefix + "1"))
            stack.append((node[2], prefix + "0"))

    return code_dict

def mean_length(code_dict, probs_dict) -> float:
    """
    Calculates the mean code length of a Huffman code.
    
    Parameters:
        code_dict: dict {symbol: code}
        probs_dict: dict {symbol: probability}
    
    Returns:
        float: expected number of bits per symbol
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

# TP2

def gray2binary(bits):
    """"
    Converts a vector of bits in Gray code to binary code.

    Parameters:
        bits: np.array, the input vector of bits in Gray code
    Returns:
        np.array: the output vector of bits in binary code
    """
    binary = np.zeros_like(bits)
    binary[0] = bits[0]
    for i in range(1, len(bits)):
        binary[i] = binary[i-1] ^ bits[i]

    return binary

def binary2decimal(bits):
    """
    Converts a vector of bits to a decimal number.

    Parameters:
        bits: np.array, the input vector of bits
    Returns:
        int: the decimal number
    """
    return int(''.join(map(str, bits)), 2)

def amplitude_label(bits_axis, n_levels, code_label):
    if len(bits_axis) == 0:
        return 0

    if code_label == "Gray":
        bits_axis = gray2binary(bits_axis)
    pos = binary2decimal(bits_axis)
    return 2 * pos - (n_levels - 1)

def modulation(char_list, modulation_type, M, code_label):

    supported_types = ["QAM", "FSK"]
    if modulation_type not in supported_types:
        raise ValueError(f"Invalid modulation type. Expected one of {supported_types}, got {modulation_type}")

    if modulation_type == "FSK" and code_label == "Gray":
        raise ValueError("Gray code is not applicable for FSK modulation.")

    if M < 2 or M > 16 or np.log2(M) % 1 != 0:
        raise ValueError("M must be a power of 2 between 2 and 16 inclusive.")

    labels = ["Gray", "Binary"]
    if code_label not in labels:
        raise ValueError(f"Invalid code label. Expected one of {labels}, got {code_label}")

    bits = np.array([int(b) for b in ''.join(char_list)]) # Vector of bits (it can be done outside)

    Eb = 1                  # Energy per bit
    k = int(np.log2(M))     # Bits per symbol
    Es = Eb * k             # Energy per symbol

    if len(bits) % k != 0: # Vector length must be a multiple of k
        bits = np.pad(bits, (0, k - (len(bits) % k)), mode='constant')

    symbols = bits.reshape(-1, k) # (N words x k bits)
    symbols_idx = np.array([binary2decimal(row) for row in symbols]) # decimal representation of the N words (N x 1)

    if modulation_type == "QAM":
        kI, kQ = np.ceil(k / 2), np.floor(k / 2) # Round up & down
        MI, MQ = 2 ** kI, 2 ** kQ

        constellation = np.zeros((M, 2))
        for number in range(M):
            binary_number = np.array(list(np.binary_repr(number, width=k)), dtype=int) # decimal to binary
            coordI = amplitude_label(binary_number[:kI], MI, code_label)
            coordQ = amplitude_label(binary_number[kI:], MQ, code_label)
            constellation[number] = [coordI, coordQ]

        Es_grid = np.mean(np.sum(constellation**2, axis=1))
        constellation *= np.sqrt(Es / Es_grid)

        return constellation[symbols_idx] # (N x 2)

    if modulation_type == "FSK":
        coords = np.zeros((len(symbols), M))
        coords[np.arange(len(symbols)), symbols_idx] = np.sqrt(Es)

        return coords # (N x M)
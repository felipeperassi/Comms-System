import numpy as np

from scripts.transmitter import amplitude_label
from scripts.transmitter import encode_block

# ----------------------------------- TP1 -----------------------------------

def decode_text(codified_codes, code_dict) -> list:
    """
    Decodes a list of codified binary codes using the provided code dictionary.
    
    Parameters:
        codified_codes: list of str, the codified binary codes to be decoded
        code_dict: dict {symbol: code}
    
    Returns:
        list: the decoded symbols corresponding to the codified codes
    """
    reverse_code_dict = {code: symbol for symbol, code in code_dict.items()} # reverse mapping from code to symbol
    return [reverse_code_dict[code] for code in codified_codes]

def huffman_decode(binary_vector, code_dict) -> list:
    """
    Decodes a continuous bit vector into symbols using a Huffman code dictionary,
    exploiting the prefix-free property to find the codeword boundaries.

    Parameters:
        binary_vector: np.array, the continuous stream of bits (0/1)
        code_dict: dict {symbol: codeword string}

    Returns:
        list: the decoded symbols
    """
    reverse_code_dict = {code: symbol for symbol, code in code_dict.items()}

    decoded = []
    current = ""
    for bit in binary_vector:
        current += str(int(bit))
        if current in reverse_code_dict:
            decoded.append(reverse_code_dict[current])
            current = ""

    return decoded

def write_file(filename, decoded_list) -> None:
    """
    Writes a list of decoded symbols to a file.

    Parameters:
        filename: str, the name of the file to write to
        decoded_list: list, the list of decoded symbols to be written to the file
    """
    content = ''.join(decoded_list)
    with open(filename, 'w',encoding='utf-8') as f:
        f.write(content)

# ----------------------------------- TP2 -----------------------------------

def decimal2binary(decimal_number, width) -> np.array:
    """
    Converts a decimal number to a binary vector with a fixed width.

    Parameters:
        decimal_number: int, the decimal number to convert
        width: int, the number of bits in the output vector

    Returns:
        np.array: the binary representation as a vector of 0s and 1s
    """
    return np.array(list(np.binary_repr(decimal_number, width=width)), dtype=int)

def demodulate_symbols(received_signal, modulation_type, M, code_label, original_length=None) -> np.array:
    """
    Demodulates a received signal and recovers the transmitted binary vector.

    Parameters:
        received_signal: np.array, the received signal (N x 2 for QAM or N x M for FSK)
        modulation_type: str, the type of modulation ("QAM" or "FSK")
        M: int, the number of symbols in the constellation
        code_label: str, the type of code ("Gray" or "Binary")
        original_length: int, optional original bit length to remove transmitter padding

    Returns:
        np.array: demodulated binary vector
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

    k = int(np.log2(M))
    Eb = 1
    Es = Eb * k
    received_signal = np.asarray(received_signal)

    if modulation_type == "QAM":
        if received_signal.ndim != 2 or received_signal.shape[1] != 2:
            raise ValueError("QAM received signal must have shape (N, 2).")

        kI, kQ = int(np.ceil(k / 2)), int(np.floor(k / 2))
        MI, MQ = 2 ** kI, 2 ** kQ

        constellation = np.zeros((M, 2))
        for number in range(M):
            binary_number = decimal2binary(number, k)
            coordI = amplitude_label(binary_number[:kI], MI, code_label)
            coordQ = amplitude_label(binary_number[kI:], MQ, code_label)
            constellation[number] = [coordI, coordQ]

        Es_grid = np.mean(np.sum(constellation**2, axis=1))
        constellation *= np.sqrt(Es / Es_grid)

        distances = np.linalg.norm(received_signal[:, None, :] - constellation[None, :, :], axis=2)
        symbols_idx = np.argmin(distances, axis=1)

    else:
        if received_signal.ndim != 2 or received_signal.shape[1] != M:
            raise ValueError(f"FSK received signal must have shape (N, {M}).")

        symbols_idx = np.argmax(received_signal, axis=1)

    binary_vector = np.concatenate([decimal2binary(symbol_idx, k) for symbol_idx in symbols_idx])

    if original_length is not None:
        binary_vector = binary_vector[:original_length]

    return binary_vector, symbols_idx

def symbol_error_probability(transmitted_idx, demodulated_idx) -> float:
    """
    Calculates the symbol error probability given the transmitted and demodulated symbol indices.

    Parameters:
        transmitted_idx: np.array, the indices of the transmitted symbols
        demodulated_idx: np.array, the indices of the demodulated symbols

    Returns:
        float: the symbol error probability
    """
    return np.sum(transmitted_idx != demodulated_idx) / len(transmitted_idx)

def bit_error_probability(transmitted_bits, received_bits) -> float:
    """
    Calculates the bit error probability given the transmitted and received binary vectors.
    
    Parameters:
        transmitted_bits: np.array, the original transmitted binary vector
        received_bits: np.array, the binary vector obtained after demodulation
    
    Returns:
        float: the bit error probability
    """
    return np.sum(transmitted_bits != received_bits) / len(transmitted_bits)

# ----------------------------------- TP4 -----------------------------------

def parity(G, k, n) -> np.array:
    """
    Constructs the parity-check matrix H from the generator matrix G.
    
    Parameters:
        G: np.array, the generator matrix of size (k, n)
        k: int, the number of message bits
        n: int, the total number of bits in the codeword (message + parity)
    
    Returns:
        np.array: the parity-check matrix H of size (n-k, n)
    """
    P = G[:, :n-k]  # G = [P | I(k)] 
    H = np.hstack((np.eye(n-k, dtype=int), P.T))  # H = [I | P^T]
    return H

def syndrome(U, H) -> np.array:
    """
    Calculates the syndrome of a received block U using the parity-check matrix H.
    
    Parameters:
        U: np.array, the received block of n bits
        
    Returns:
        np.array: the syndrome of U
    """
    return tuple(np.mod(U @ H.T, 2).astype(int).tolist()) # To tuple for hashability in syndrome table

def syndrome_table(H, n) -> dict:
    """
    Constructs a syndrome table for single-bit error patterns based on the parity-check matrix H.
    
    Parameters:
        H: np.array, the parity-check matrix of size (n-k, n)
        n: int, the total number of bits in the codeword (message + parity)
    
    Returns:
        dict: a dictionary mapping syndromes to their corresponding single-bit error patterns
    """
    table = {}
    for i in range(n):
        error = np.zeros(n, dtype=int)  # Error pattern
        error[i] = 1                    # Error in bit i

        s = syndrome(error, H)  # Syndrome for this error pattern
        table[s] = error

    return table 

def decode_block(U, H, S, k) -> np.array:
    """
    Decodes a received block U using the parity-check matrix H and the syndrome table S.
    
    Parameters:
        U: np.array, the received block of n bits
        H: np.array, the parity-check matrix of size (n-k, n)
        S: dict, the syndrome table mapping syndromes to error patterns
        k: int, the number of message bits (the last k bits of the codeword)
    
    Returns:
        np.array: the decoded message bits (the last k bits of the corrected codeword)
    """
    s = syndrome(U, H) # Calculate the syndrome of the received block
    if s in S: # If the syndrome corresponds to a known error pattern
        U = np.mod(U + S[s], 2)

    return U[-k:] 

def decodificate_channel(binary_vector, H, S, k, n) -> np.array:
    """
    Decodes a binary vector that has been encoded with a linear block code and potentially corrupted by errors.
    
    Parameters:
        binary_vector: np.array, the received binary vector to be decoded
        H: np.array, the parity-check matrix of size (n-k, n)
        S: dict, the syndrome table mapping syndromes to error patterns
        k: int, the number of message bits (the last k bits of the codeword)
        n: int, the total number of bits in the codeword (message + parity)
    
    Returns:
        np.array: the decoded binary vector containing only the message bits
    """
    blocks = binary_vector.reshape(-1, n)
    decoded_blocks = np.array([decode_block(block, H, S, k) for block in blocks])

    return decoded_blocks.flatten()


def code_parameters(G, k, n) -> tuple:
    """
    Calculates the minimum distance dmin, the maximum number of errors that can be detected e, and the maximum number of errors that can be corrected t for a linear block code defined by its generator matrix G.
    
    Parameters:
        G: np.array, the generator matrix of size (k, n)
        k: int, the number of message bits
        n: int, the total number of bits in the codeword (message + parity)
    
    Returns:
        tuple: (dmin, e, t) where dmin is the minimum distance, e is the maximum number of errors that can be detected, and t is the maximum number of errors that can be corrected
    """
    codewords = []
    for i in range(2**k): # Generate all possible codewords by encoding all possible messages
        message = np.array(list(np.binary_repr(i, width=k)), dtype=int)
        codeword = encode_block(message, G)
        codewords.append(codeword)

    weights = [np.sum(codeword) for codeword in codewords if np.sum(codeword) > 0]

    dmin = int(min(weights))
    e = dmin - 1            # Detectable errors
    t = (dmin - 1) // 2     # Correctable errors

    return dmin, e, t
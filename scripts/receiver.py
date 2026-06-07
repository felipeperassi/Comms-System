
import numpy as np

from scripts.transmitter import amplitude_label

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

def write_file(filename, decoded_list) -> None:
    """
    Writes a list of decoded symbols to a file.

    Parameters:
        filename: str, the name of the file to write to
        decoded_list: list, the list of decoded symbols to be written to the file
    """
    content = ''.join(decoded_list)
    with open(filename, 'w') as f:
        f.write(content)

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


# TP2
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

    return binary_vector

import numpy as np

from scripts.transmitter import encode_block

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


def parity(G, k, n) -> np.array:
    P = G[:, :n-k]        #G tiene la identidad en el lado derecho     # primeras n-k columnas (la parte P)
    H = np.hstack((P.T, np.eye(n-k, dtype=int))) # H = [P^T | I(n-k)]
    return H

def syndrome(U, H) -> np.array:
    return tuple(np.mod(U @ H.T, 2).astype(int))

def syndrome_table(H, n) -> dict:

    table = {}

    for i in range(n):
        error = np.zeros(n, dtype=int)  # patrón de error
        error[i] = 1                    # error en el bit i

        s = tuple(np.mod(error @ H.T, 2).astype(int))  # síndrome
        table[s] = error

    return table #me devuelve un diccionario con el síndrome como clave y el patrón de error como valor

def decode_block(U, H, S, k) -> np.array:
    #Decodifica y corrige una palabra de n bits.

    s = syndrome(U, H)  # calculo el síndrome

    if s in S:
        U = np.mod(U + S[s], 2)  # corrijo el error

    return U[-k:]  # devuelvo los últimos k bits (mensaje original)

def decodificate_channel(binary_vector, H, S, k, n) -> np.array:
    
    #Organiza el vector binario en bloques de n bits y decodifica cada uno.

    blocks = binary_vector.reshape(-1, n)
    decoded_blocks = np.array([decode_block(block, H, S, k) for block in blocks])

    return decoded_blocks.flatten()


def code_parameters(G, k, n) -> tuple:
    
    ##Calcula la distancia mínima dmin, la cantidad máxima de errores a detectar e y a corregir t.

    codewords = []
    for i in range(2**k): # genero todas las 2^k palabras código posibles
        message = np.array(list(np.binary_repr(i, width=k)), dtype=int)  # convierte i a binario de k bits ydesp convierte el string a array
        codeword = encode_block(message, k, n, G)
        codewords.append(codeword)

    # calculo el peso de cada palabra código (cantidad de 1s), sin contar la palabra cero
    weights = [np.sum(codeword) for codeword in codewords if np.sum(codeword) > 0]

    dmin = int(min(weights))
    e = dmin - 1        # errores que puede detectar
    t = (dmin - 1) // 2 # errores que puede corregir

    return dmin, e, t
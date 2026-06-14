import numpy as np
from data.config import TXT_PATH, MEDIA_PATH
from scripts.transmitter import appearence_probs, huffman_algorithm, codificate_text, codificate_channel
from scripts.receiver import decodificate_channel, parity, syndrome_table, code_parameters
from scripts.extras import print_dict

def channel_coding(binary_vector):

    # Given parameters
    k = 4
    n = 8
    G = np.array([
        [1, 1, 1, 0, 1, 0, 0, 0],
        [1, 1, 0, 1, 0, 1, 0, 0],
        [1, 0, 1, 1, 0, 0, 1, 0],
        [0, 1, 1, 1, 0, 0, 0, 1]
    ], dtype=int)

    print(f"Generator matrix G:\n{G}", f"Message length (k): {k}", f"Codeword length (n): {n}", sep="\n")

    # Calculate parity-check matrix H
    H = parity(G, k, n)
    print(f"\nParity-check matrix H:\n{H}")

    # Calculate parameters of the code
    dmin, e, t = code_parameters(G, k, n)
    print(f"\nMinimum distance (dmin): {dmin}", f"Detectable errors (e): {e}", f"Correctable errors (t): {t}", sep="\n")

    # Create syndrome table
    S = syndrome_table(H, n)
    print_dict(S, title="\nSyndrome Table (Syndrome -> Error Pattern)")

    # Corrupted codeword example
    binary_vector = binary_vector[:k]  # Take only one word with the first k bits of the input vector
    print(f"\nOriginal binary vector (message bits only): {binary_vector}")
    
    encoded_vector = codificate_channel(binary_vector, G, k, n)
    error_vector = encoded_vector.copy()
    error_vector[0] = (error_vector[0] + 1) % 2  # Introduce an error in the first bit
    print(f"Encoded vector: {encoded_vector}", f"Error vector (with one bit flipped): {error_vector}", sep="\n")

    decoded_vector = decodificate_channel(error_vector, H, S, k, n)
    print(f"Decoded vector (after error correction): {decoded_vector}")

if __name__ == "__main__":
    with open(TXT_PATH, 'r') as f:
        text = f.read()

    # Source codification
    probs_dict, char_counts_dict = appearence_probs(text)
    code_dict = huffman_algorithm(probs_dict)
    codified_text = codificate_text(text, code_dict) # ['10011100', '1111111', ... ] for each char in the text 

    # Adapt the output of codification to be a binary vector (list of 0s and 1s)
    binary_vector = np.array([int(bit) for symbol in codified_text for bit in symbol]) # [1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, ... ]

    # Test channel encoding & error correction
    channel_coding(binary_vector)

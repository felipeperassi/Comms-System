import numpy as np
from data.config import TXT_PATH, OUTPUT_PATH, MEDIA_PATH
from scripts.transmitter import (
    appearence_probs, entropy, huffman_algorithm, mean_length, minimum_length, shannon_range, codificate_text, 
    modulate_symbols, calculate_energies,codificate_channel
)
from scripts.receiver import decode_text, write_file, code_parameters, decodificate_channel, syndrome_table, parity, demodulate_symbols, symbol_error_probability, bit_error_probability
from scripts.extras import plot_char_counts, print_dict, plot_constellation
from scripts.channel import channel_effects

def transmitter_pass(text, channel_coding=False, k=None, n=None, G=None, modulation_type="QAM", M=16, code_label="Binary"):

    # Source codification
    probs_dict, char_counts_dict = appearence_probs(text)
    code_dict = huffman_algorithm(probs_dict)
    codified_text = codificate_text(text, code_dict) # ['10011100', '1111111', ... ] for each char in the text 
  
    # Adapt the output of codification to be a binary vector (list of 0s and 1s)
    binary_vector = np.array([int(bit) for symbol in codified_text for bit in symbol]) 

    # Channel coding (if enabled)
    if channel_coding and (G is None or k is None or n is None):
        raise ValueError("channel_coding=True requires G, k and n.")
    
    encoded_vector = codificate_channel(binary_vector, G, k, n) if channel_coding else binary_vector

    # Modulation
    mod_symbols, mod_symbol_idxs = modulate_symbols(encoded_vector, modulation_type=modulation_type, M=M, code_label=code_label)

    return binary_vector, mod_symbols, mod_symbol_idxs

def receiver_pass(channel_symbols, binary_vector_length, channel_coding=False, k=None, n=None, G=None, modulation_type="QAM", M=16, code_label="Binary"):

    # Channel coding parameters (if enabled)
    if channel_coding and (G is None or k is None or n is None):
        raise ValueError("channel_coding=True requires G, k and n.")
    
    # Demodulation
    demod_symbols, demod_symbol_idxs = demodulate_symbols(channel_symbols, modulation_type=modulation_type, M=M, code_label=code_label, original_length=binary_vector_length)

    # Channel decoding (if enabled)
    H = parity(G, k, n)
    S = syndrome_table(H, n)
    decoded_vector = decodificate_channel(demod_symbols, H, S, k, n) if channel_coding else demod_symbols

    return decoded_vector, demod_symbols

if __name__ == "__main__":
    with open(TXT_PATH, 'r') as f:
        text = f.read()
    
    # Given parameters
    k = 4
    n = 8
    G = np.array([
        [1, 1, 1, 0, 1, 0, 0, 0],
        [1, 1, 0, 1, 0, 1, 0, 0],
        [1, 0, 1, 1, 0, 0, 1, 0],
        [0, 1, 1, 1, 0, 0, 0, 1]
    ], dtype=int)

    # Vectors for performance evaluation
    EbN0_vec = np.arange(0, 11, 1)  
    modulation_types = ["QAM", "FSK"]
    M_vec = [2, 4, 8, 16]

    symbol_error = []
    bit_error = []

    # Loop for evaluating performance across different EbN0s, modulation types and constellation sizes
    for EbN0 in EbN0_vec:
        for modulation_type in modulation_types:
            for M in M_vec:
                print(f"----- Evaluating {modulation_type} with M={M} at EbN0={EbN0} dB -----")

                # Transmitter pass
                binary_vector, mod_symbols, mod_symbol_idxs = transmitter_pass(text, channel_coding=True, k=k, n=n, G=G, modulation_type=modulation_type, M=M, code_label="Binary")

                # Channel effects
                Eb = 1  # Energy per bit
                EbN0_linear = 10 ** (EbN0 / 10)  # Convert dB to linear scale
                N_0 = Eb / EbN0_linear

                channel_symbols = channel_effects(mod_symbols, N_0)

                # Receiver pass
                decoded_vector, demod_symbols_idx = receiver_pass(channel_symbols, len(binary_vector), channel_coding=True, k=k, n=n, G=G, modulation_type=modulation_type, M=M, code_label="Binary")

                # Error evaluation
                Pe = symbol_error_probability(mod_symbol_idxs, demod_symbols_idx)
                Pb = bit_error_probability(binary_vector, decoded_vector)
                print(f"SNR: {EbN0} dB - Pe: {Pe:.4f}, Pb: {Pb:.4f}")

                symbol_error.append((EbN0, modulation_type, M, Pe)), bit_error.append((EbN0, modulation_type, M, Pb))
            
    # Plotting results
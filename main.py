import numpy as np
from scipy.special import erfc
from data.config import TXT_PATH, OUTPUT_PATH
from scripts.transmitter import appearence_probs, huffman_algorithm, codificate_text, modulate_symbols, codificate_channel
from scripts.receiver import huffman_decode, write_file, decodificate_channel, syndrome_table, parity, demodulate_symbols, symbol_error_probability, bit_error_probability
from scripts.extras import plot_error_curve
from scripts.channel import channel_effects, awgn

def q_function(x) -> np.ndarray:
    """
    Q-function, defined as Q(x) = 0.5 * erfc(x / sqrt(2))

    Parameters:
        x (float or np.ndarray): Input value(s) for which to compute the Q-function.
    
    Returns:
        np.ndarray: The computed Q-function values for the input x.
    """
    return 0.5 * erfc(x / np.sqrt(2))


def theo_error_probabilities(modulation_type, M, EbN0_dB) -> tuple:
    """
    Calculate the theoretical symbol error probability (Pe) and bit error probability (Pb) for a given modulation type, constellation size, and Eb/N0.
    
    Parameters:
        modulation_type (str): The type of modulation ("QAM" or "FSK").
        M (int): The constellation size (number of symbols).
        EbN0_dB (float): The energy per bit to noise power spectral density ratio in decibels.
    
    Returns:
        tuple: A tuple containing the symbol error probability (Pe) and bit error probability (Pb).
    """
    EbN0 = 10 ** (EbN0_dB / 10)
    k = np.log2(M)

    if modulation_type == "QAM":
        if M == 2:                                   
            Pe = q_function(np.sqrt(2 * EbN0))
            return Pe, Pe
        
        p = 2 * (1 - 1 / np.sqrt(M)) * q_function(np.sqrt(3 * k / (M - 1) * EbN0))
        Pe = 1 - (1 - p) ** 2
        return Pe, Pe / k                         

    if modulation_type == "FSK":
        Pe = (M - 1) * q_function(np.sqrt(k * EbN0)) 
        return Pe, Pe * 2 ** (k - 1) / (2 ** k - 1)

    raise ValueError(f"Invalid modulation type. Expected 'QAM' or 'FSK', got {modulation_type}")


def transmitter_pass(text, channel_coding=False, k=None, n=None, G=None, modulation_type="QAM", M=16, code_label="Binary"):
    """
    Performs the transmitter operations: source codification, channel coding (if enabled) and modulation, returning the binary vector, encoded vector (if channel coding is enabled), modulated symbols and their corresponding indices for error evaluation.
    
    Parameters:
        text: str, the input text to be transmitted
        channel_coding: bool, whether to apply channel coding (default: False)
        k: int, the number of information bits per codeword (required if channel_coding=True)
        n: int, the total number of bits per codeword (required if channel_coding=True)
        G: np.array of shape (k, n), the generator matrix for channel coding (required if channel_coding=True)
        modulation_type: str, the type of modulation to use ("QAM" or "FSK", default: "QAM")
        M: int, the constellation size for modulation (e.g., M=16 for 16-QAM, default: 16)
        code_label: str, the type of code for constellation labeling ("Gray" or "Binary", default: "Binary")

    Returns:
        binary_vector: np.array of shape (N,) containing the binary representation of the codified text before channel coding
        encoded_vector: np.array of shape (N_encoded,) containing the binary vector after channel coding (if enabled), otherwise same as binary_vector
        mod_symbols: np.array of shape (N_mod, 2) containing the I/Q coordinates of the modulated symbols
        mod_symbol_idxs: np.array of shape (N_mod,) containing the indices of the modulated symbols in the constellation, used for error evaluation against the demodulated symbol indices at the receiver
    """
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

    return binary_vector, encoded_vector, mod_symbols, mod_symbol_idxs, code_dict

def receiver_pass(channel_symbols, binary_vector_length, code_dict, channel_coding=False, k=None, n=None, G=None, modulation_type="QAM", M=16, code_label="Binary"):
    """
    Performs the receiver operations: demodulation, channel decoding (if enabled) and returns the decoded binary vector and the demodulated symbol indices for error evaluation.
    
    Parameters:
        channel_symbols: np.array of shape (N, 2) representing the received I/Q symbols after channel effects
        binary_vector_length: int, the length of the original binary vector before channel coding (used to correctly demodulate only the relevant symbols)
        channel_coding: bool, whether channel coding was used in the transmitter
        k: int, the number of information bits per codeword (required if channel_coding=True)
        n: int, the total number of bits per codeword (required if channel_coding=True)
        G: np.array of shape (k, n), the generator matrix used for channel coding (required if channel_coding=True)
        modulation_type: str, the type of modulation used ("QAM" or "FSK")
        M: int, the constellation size used in modulation (e.g., M=16 for 16-QAM)
        code_label: str, the type of code used for constellation labeling ("Gray" or "Binary")

    Returns:
        decoded_vector: np.array of shape (binary_vector_length,) containing the decoded binary bits after demodulation and channel decoding (if enabled)
        demod_symbol_idxs: np.array of shape (N,) containing the indices of the demodulated symbols, used for error evaluation against the original mod_symbol_idxs from the transmitter
    """
    # Channel coding parameters (if enabled)
    if channel_coding and (G is None or k is None or n is None):
        raise ValueError("channel_coding=True requires G, k and n.")
    
    # Demodulation
    demod_symbols, demod_symbol_idxs = demodulate_symbols(channel_symbols, modulation_type=modulation_type, M=M, code_label=code_label, original_length=binary_vector_length)

    # Channel decoding (if enabled)
    if channel_coding:
        H = parity(G, k, n)
        S = syndrome_table(H, n)
        decoded_vector = decodificate_channel(demod_symbols, H, S, k, n)
    else: 
        decoded_vector = demod_symbols
    
    # Source decoding
    decoded_text = huffman_decode(decoded_vector, code_dict) if code_dict is not None else None
        
    return decoded_vector, demod_symbol_idxs, decoded_text

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
    channel_coding = True # Set to True to enable channel coding in the performance evaluation loop

    # Loop for evaluating performance across different EbN0s, modulation types and constellation sizes
    for modulation_type in modulation_types:
        for M in M_vec:

            sim_Pe, sim_Pb, theo_Pe, theo_Pb = [], [], [], []
            for EbN0 in EbN0_vec:
                print(f"----- Evaluating {modulation_type} with M={M} at EbN0={EbN0} dB -----")

                # Transmitter pass
                binary_vector, encoded_vector, mod_symbols, mod_symbol_idxs, code_dict = transmitter_pass(text, channel_coding=channel_coding, k=k, n=n, G=G, 
                                                                                               modulation_type=modulation_type, M=M, code_label="Binary")

                # Channel effects
                Eb = 1  # Energy per bit
                EbN0_linear = 10 ** (EbN0 / 10)  # Convert dB to linear scale
                N_0 = Eb / EbN0_linear

                channel_symbols = mod_symbols + awgn(mod_symbols.shape, N_0)
                # channel_symbols = channel_effects(mod_symbols, N_0) # Uncomment to include attenuation effects in the channel.

                # Receiver pass
                decoded_vector, demod_symbol_idxs, decoded_text = receiver_pass(channel_symbols, len(encoded_vector), code_dict, channel_coding=channel_coding, k=k, n=n, G=G, 
                                                                  modulation_type=modulation_type, M=M, code_label="Binary")

                # Output decoded text
                output_filename = f"decoded_{modulation_type}_M{M}_EbN0{EbN0}dB_channel_{'coded' if channel_coding else 'uncoded'}.txt"             
                write_file(OUTPUT_PATH / output_filename, decoded_text) if decoded_text is not None else None

                # Error evaluation
                sym_error_sim = symbol_error_probability(mod_symbol_idxs, demod_symbol_idxs)
                bit_error_sim = bit_error_probability(binary_vector, decoded_vector[:len(binary_vector)])
                sim_Pe.append(sym_error_sim), sim_Pb.append(bit_error_sim)
                print(f"SNR: {EbN0} dB - Pe: {sym_error_sim:.4f}, Pb: {bit_error_sim:.4f}")
                
                # Theoretical error probabilities
                sym_error_theo, bit_error_theo = theo_error_probabilities(modulation_type, M, EbN0)
                theo_Pe.append(sym_error_theo), theo_Pb.append(bit_error_theo)
                print(f"SNR: {EbN0} dB - Theoretical Pe: {sym_error_theo:.4f}, Theoretical Pb: {bit_error_theo:.4f}")
            
            plot_error_curve(EbN0_vec, theo_Pb, sim_Pb, title=f"Bit Error Probability - {modulation_type} - M={M}", 
                             y_label=r"$P_b$", filename=f"bep_{modulation_type}_{M}_channel_{channel_coding}.png")
            plot_error_curve(EbN0_vec, theo_Pe, sim_Pe, title=f"Symbol Error Probability - {modulation_type} - M={M}", 
                             y_label=r"$P_e$", filename=f"sep_{modulation_type}_{M}_channel_{channel_coding}.png")
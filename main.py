import numpy as np
from scipy.special import erfc
from data.config import MAX_TEXT_CHARS, TXT_PATH, OUTPUT_PATH
from scripts.transmitter import appearence_probs, huffman_algorithm, codificate_text, modulate_symbols, codificate_channel
from scripts.receiver import huffman_decode, write_file, decodificate_channel, syndrome_table, parity, demodulate_symbols, symbol_error_probability, bit_error_probability
from scripts.extras import plot_error_curve
from scripts.channel import channel_effects, awgn

def read_simulation_text(path, max_chars=None):
    """
    Reads a bounded UTF-8 text sample for the simulation.

    The transmitter builds several full-size arrays from this text, so using a
    huge input file directly can require many times the file size in RAM.
    """
    with open(path, 'r', encoding='utf-8') as f:
        return f.read() if max_chars is None else f.read(max_chars)

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
        # if M == 2:                                   
        #     Pe = q_function(np.sqrt(2 * EbN0))
        #     return Pe, Pe
        
        # p = 2 * (1 - 1 / np.sqrt(M)) * q_function(np.sqrt(3 * k / (M - 1) * EbN0))
        # Pe = 1 - (1 - p) ** 2
        kI, kQ = int(np.ceil(k / 2)), int(np.floor(k / 2))
        MI, MQ = 2 ** kI, 2 ** kQ

        Es_grid = ((MI ** 2 - 1) + (MQ ** 2 - 1)) / 3
        q_arg = np.sqrt(2 * k * EbN0 / Es_grid)
        q = q_function(q_arg)

        Pc_I = 1 - 2 * (1 - 1 / MI) * q
        Pc_Q = 1 - 2 * (1 - 1 / MQ) * q if MQ > 1 else 1

        Pe = 1 - Pc_I * Pc_Q
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
        code_label: str, the type of code for constellation labeling ("Binary" or "Binary", default: "Binary")

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
        code_label: str, the type of code used for constellation labeling ("Binary" or "Binary")

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
    text = read_simulation_text(TXT_PATH, MAX_TEXT_CHARS)

    text_length = len(text)
    print(f"Longitud del texto: {text_length}")
    
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
    modulation_types = ["QAM","FSK"]
    M_vec = [2, 4, 8, 16]
    channel_coding = True # Set to True to enable channel coding in the performance evaluation loop

    # Loop for evaluating performance across different EbN0s, modulation types and constellation sizes
    for modulation_type in modulation_types:
        for M in M_vec:

            sim_Pe_cod, sim_Pb_cod, theo_Pe_cod, theo_Pb_cod = [], [], [], []
            sim_Pe_uncod, sim_Pb_uncod, theo_Pe_uncod, theo_Pb_uncod = [], [], [], []
            for EbN0 in EbN0_vec:
                print(f"----- Evaluating {modulation_type} with M={M} at EbN0={EbN0} dB -----")

                # Transmitter pass coded
                binary_vector, encoded_vector, mod_symbols, mod_symbol_idxs, code_dict = transmitter_pass(text, channel_coding=True, k=k, n=n, G=G, 
                                                                                               modulation_type=modulation_type, M=M, code_label="Binary")
                
                # Transmitter pass uncoded 
                binary_vector_uncoded, _, mod_symbols_uncoded, mod_symbol_idxs_uncoded, _ = transmitter_pass(text, channel_coding=False, modulation_type=modulation_type, M=M, code_label="Binary")

                # Channel effects
                #Eb_cod = 1  * (n / k)   # Energy per bit coded
                Eb_cod = 1  * (k / n)   # Energy per bit coded
                Eb_uncod = 1            # Energy per bit uncoded (same as coded for fair comparison)
                EbN0_linear = 10 ** (EbN0 / 10)  # Convert dB to linear scale
                N_0_uncod = Eb_uncod / EbN0_linear
                N_0_cod = Eb_cod / EbN0_linear
               
                channel_symbols_cod = mod_symbols + awgn(mod_symbols.shape, N_0_cod)
                channel_symbols_uncod = mod_symbols_uncoded + awgn(mod_symbols_uncoded.shape, N_0_uncod)
                # channel_symbols = channel_effects(mod_symbols, N_0) # Uncomment to include attenuation effects in the channel.

                # Receiver pass coded
                decoded_vector, demod_symbol_idxs, decoded_text = receiver_pass(channel_symbols_cod, len(encoded_vector), code_dict, channel_coding=True, k=k, n=n, G=G, 
                                                                  modulation_type=modulation_type, M=M, code_label="Binary")

                # Receiver pass uncoded
                decoded_vector_uncoded, demod_symbol_idxs_uncoded, decoded_text_uncoded = receiver_pass(channel_symbols_uncod, len(binary_vector_uncoded), code_dict, channel_coding=False, k=k, n=n, G=G, 
                                                                  modulation_type=modulation_type, M=M, code_label="Binary")
                # Output decoded text
                output_filename_coded = f"decoded_{modulation_type}_M{M}_EbN0{EbN0}dB_channel_coded.txt"
                write_file(OUTPUT_PATH / output_filename_coded, decoded_text) if decoded_text is not None else None

                output_filename_uncoded = f"decoded_{modulation_type}_M{M}_EbN0{EbN0}dB_channel_uncoded.txt"
                write_file(OUTPUT_PATH / output_filename_uncoded, decoded_text_uncoded) if decoded_text_uncoded is not None else None

                # Error evaluation
                sym_error_cod = symbol_error_probability(mod_symbol_idxs, demod_symbol_idxs)
                bit_error_cod = bit_error_probability(binary_vector, decoded_vector[:len(binary_vector)])
                sym_error_uncod = symbol_error_probability(mod_symbol_idxs_uncoded, demod_symbol_idxs_uncoded)
                bit_error_uncod = bit_error_probability(binary_vector_uncoded, decoded_vector_uncoded[:len(binary_vector_uncoded)])
                
                sim_Pe_cod.append(sym_error_cod), sim_Pb_cod.append(bit_error_cod)
                sim_Pe_uncod.append(sym_error_uncod), sim_Pb_uncod.append(bit_error_uncod)

                print(f"SNR: {EbN0} dB - Pe: {sym_error_cod:.4f}, Pb: {bit_error_cod:.4f}")
                print(f"SNR: {EbN0} dB - Pe (uncoded): {sym_error_uncod:.4f}, Pb (uncoded): {bit_error_uncod:.4f}")

                # Theoretical error probabilities
                sym_error_theo, bit_error_theo = theo_error_probabilities(modulation_type, M, EbN0)
                theo_Pe_cod.append(sym_error_theo), theo_Pb_cod.append(bit_error_theo)
                print(f"SNR: {EbN0} dB - Theoretical Pe: {sym_error_theo:.4f}, Theoretical Pb: {bit_error_theo:.4f}")
            
            plot_error_curve(EbN0_vec, theo_error=theo_Pb_cod, sim_error_cod=sim_Pb_cod, sim_error_uncod=sim_Pb_uncod, title=f"Bit Error Probability - {modulation_type} - M={M}", 
                             y_label=r"$P_b$", filename=f"bep_{modulation_type}_{M}_channel_comp.png")
            plot_error_curve(EbN0_vec, theo_error=theo_Pe_cod, sim_error_cod=sim_Pe_cod, sim_error_uncod=sim_Pe_uncod, title=f"Symbol Error Probability - {modulation_type} - M={M}", 
                             y_label=r"$P_e$", filename=f"sep_{modulation_type}_{M}_channel_comp.png")

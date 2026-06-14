import numpy as np
from data.config import PATH, OUTPUT_PATH, MEDIA_PATH
from scripts.transmitter import (
    appearence_probs, entropy, huffman_algorithm, mean_length, minimum_length, shannon_range, codificate_text, 
    modulate_symbols, calculate_energies,codificate_channel
)
from scripts.receiver import decode_text, write_file, code_parameters, decodificate_channel, syndrome_table, parity, demodulate_symbols, symbol_error_probability, bit_error_probability
from scripts.extras import plot_char_counts, print_dict, plot_constellation
from scripts.channel import channel_effects

#CODIFICACION DE FUENTE

def codificacion_fuente(text, return_codified=False):
    # Calculate probabilities & counts
    probs_dict, char_counts_dict = appearence_probs(text)
    print_dict(char_counts_dict, "Character counts:", sort=True)
    print_dict(probs_dict, "Appearance probabilities:", sort=True)
    plot_char_counts(char_counts_dict, MEDIA_PATH)
    
    # Huffman coding
    code_dict = huffman_algorithm(probs_dict)
    print_dict(code_dict, "Huffman codes:", sort=True)

    # Calculate entropy, Shannon range, mean and minimum code lengths
    entropy_value = entropy(probs_dict)
    shannon_rng = shannon_range(entropy_value)
    print(f"Entropy: {entropy_value:.3f} bits/symbol",
        f"Shannon range: {shannon_rng[0]:.3f} to {shannon_rng[1]:.3f} bits/symbol", sep="\n")

    mean_len, min_len = mean_length(code_dict, probs_dict), minimum_length(code_dict)
    print(f"Mean code length: {mean_len:.3f} bits/symbol", 
           f"Minimum code length: {min_len:.3f} bits/symbol", 
           f"Efficiency: {entropy_value / mean_len:.3f}", sep="\n")

    # Codification and decoding
    codified_text = codificate_text(text, code_dict)
    decoded_text = decode_text(codified_text, code_dict)
    write_file(OUTPUT_PATH / "decoded_output.txt", decoded_text)

    # Sentence to test the codification and decoding
    text_sentence = text.split('.')[1][1::] + '.' # take the first sentence of the text, removing the leading space and adding the dot at the end
    codified_sentence = codificate_text(text_sentence, code_dict)
    decoded_sentence = decode_text(codified_sentence, code_dict)
    print(f"\nOriginal sentence: {text_sentence}", 
          f"Codified sentence: {' '.join(codified_sentence)}", 
          f"Decoded sentence: {''.join(decoded_sentence)}", sep="\n")
    
 
    if return_codified: return codified_text

#CODIFICACION DE CANAL

def codificacion_canal(binary_vector):
    
    # Parámetros del código
    k = 4
    n = 8
    G = np.array([
        [1, 1, 1, 0, 1, 0, 0, 0],
        [1, 1, 0, 1, 0, 1, 0, 0],
        [1, 0, 1, 1, 0, 0, 1, 0],
        [0, 1, 1, 1, 0, 0, 0, 1]
    ], dtype=int)

    # Transmisor
    encoded_vector = codificate_channel(binary_vector, G, k, n)

    # Receptor
    H = parity(G, k, n)
    S = syndrome_table(H, n)
    decoded_vector = decodificate_channel(encoded_vector, H, S, k, n)
    dmin, e, t = code_parameters(G, k, n)

    print(f"H:\n{H}")
    print(f"Syndrome table:\n{S}")
    print(f"dmin: {dmin}", f"e: {e}", f"t: {t}", sep="\n")

    return encoded_vector

#MODULACION

def modulacion(binary_vector):

    # Modulation
    #modulation_type, M, code_label = "QAM", 16, "Binary"
    modulation_type, M, code_label = "FSK", 2, "Binary"
    symbols, transmitted_idx = modulate_symbols(binary_vector, modulation_type, M, code_label)   
    print(f"\nConstellation points for {modulation_type} modulation with M={M} and {code_label} code:")
    print(symbols)

    # Calculate energy
    Es, Eb = calculate_energies(symbols, M)
    Eb_mean = np.mean(Eb)
    print(f"\nEnergy per bit (Eb): {Eb}", f"Energy per symbol (Es): {Es}", sep="\n")

    # Channel effects
    N0 = 0.157 # Noise power spectral density
    Eb_N0_linear = Eb_mean / N0
    Eb_N0_dB = 10 * np.log10(Eb_N0_linear)
    print(f"\nEb/N0 = {Eb_N0_dB:.2f} dB")

    received_symbols = channel_effects(symbols, N0)


    # Demodulation sin ruido
    demodulated_binary_vector = demodulate_symbols(symbols, modulation_type, M, code_label, original_length=len(binary_vector))
    print(f"\nDemodulation successful: {np.array_equal(binary_vector, demodulated_binary_vector)}")

    # Plot constellation (Transmitted)
    plot_constellation(modulation_type, symbols, MEDIA_PATH, filename=f"{modulation_type}_modulate_constellation.png")

    # demodulation con ruido
    demodulated_binary_vector_noise , demodulated_idx_noise  = demodulate_symbols(received_symbols, modulation_type, M, code_label, original_length=len(binary_vector))
    print(f"\nDemodulation successful: {np.array_equal(binary_vector, demodulated_binary_vector)}") 

    # Plot constellation (Received)
    plot_constellation(modulation_type, received_symbols, MEDIA_PATH, filename=f"{modulation_type}_received_constellation.png")

    Pe_simbolo = symbol_error_probability(transmitted_idx, demodulated_idx_noise)
    Pb_bit = bit_error_probability(binary_vector, demodulated_binary_vector_noise)

    print(f"\nProbabilidad de error de símbolo (Pe): {Pe_simbolo:.6f}")
    print(f"Probabilidad de error de bit (Pb): {Pb_bit:.6f}")


if __name__ == "__main__":
    with open(PATH, 'r') as f:
        text = f.read()
    
    codified_text = codificacion_fuente(text, return_codified=True)
    binary_vector = np.array([int(b) for b in ''.join(codified_text)]) # Vector of bits representing the codified text
    encoded_vector = codificacion_canal(binary_vector)
    modulacion(encoded_vector)

    
    


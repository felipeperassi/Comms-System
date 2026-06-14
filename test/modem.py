import numpy as np
from data.config import TXT_PATH, MEDIA_PATH
from scripts.transmitter import appearence_probs, huffman_algorithm, codificate_text, modulate_symbols, calculate_mean_energies
from scripts.receiver import demodulate_symbols
from scripts.extras import plot_constellation

def modem(binary_vector, modulation_type="QAM", M=16, code_label="Binary"):

    print(f"\n ------ Modulation type: {modulation_type}, M: {M}, Code label: {code_label} ------")

    # Modulation
    mod_symbols, mod_symbol_idxs = modulate_symbols(binary_vector, modulation_type, M, code_label)
    print(f"\nConstellation points for {modulation_type} modulation with M={M} and {code_label} code:", mod_symbols, sep="\n")

    # Generate constellation plot
    plot_constellation(modulation_type, mod_symbols, M, MEDIA_PATH, filename=f"{modulation_type}_{M}_{code_label}_constellation.png")

    # Calculate mean energies
    Eb_theo = 1
    Es_theo = Eb_theo * np.log2(M)

    Es_mean, Eb_mean = calculate_mean_energies(mod_symbols, M)

    print(f"\nMean energy per bit (Eb): {Eb_mean:.3f} (theoretical: {Eb_theo:.3f})",
            f"Mean energy per symbol (Es): {Es_mean:.3f} (theoretical: {Es_theo:.3f})", sep="\n")
     
    # Demodulation without noise & error checking
    demod_symbols, demod_symbol_idxs = demodulate_symbols(mod_symbols, modulation_type, M, code_label, original_length=len(binary_vector))
    if np.array_equal(mod_symbol_idxs, demod_symbol_idxs) and np.array_equal(binary_vector, demod_symbols):
        print(f"\nDemodulation successful: All symbols correctly demodulated.")
    else:
        num_errors = np.sum(mod_symbol_idxs != demod_symbol_idxs)
        total_symbols = len(mod_symbol_idxs)
        print(f"\nDemodulation had errors: {num_errors} out of {total_symbols} symbols were incorrectly demodulated.")

if __name__ == "__main__":
    with open(TXT_PATH, 'r') as f:
        text = f.read()

    # Source codification
    probs_dict, char_counts_dict = appearence_probs(text)
    code_dict = huffman_algorithm(probs_dict)
    codified_text = codificate_text(text, code_dict) # ['10011100', '1111111', ... ] for each char in the text 
  
    # Adapt the output of codification to be a binary vector (list of 0s and 1s)
    binary_vector = np.array([int(bit) for symbol in codified_text for bit in symbol]) # [1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, ... ]

    # Test modulation & demodulation for both modulation types
    modem(binary_vector, modulation_type="QAM", M=16, code_label="Binary")
    modem(binary_vector, modulation_type="FSK", M=2, code_label="Binary")
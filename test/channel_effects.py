import numpy as np
from data.config import TXT_PATH, MEDIA_PATH
from scripts.transmitter import appearence_probs, huffman_algorithm, codificate_text, modulate_symbols
from scripts.receiver import demodulate_symbols
from scripts.channel import channel_effects
from scripts.extras import plot_constellation

MEDIA_CHANNEL_PATH = MEDIA_PATH / "channel_effects"

def channel(binary_vector, modulation_type="QAM", M=16, code_label="Binary") -> None:
    """
    Simulates the effects of the communication channel on the transmitted symbols.
    
    Parameters:
        binary_vector: np.array, the binary vector to be transmitted
        modulation_type: str, the type of modulation ("QAM" or "FSK")
        M: int, the number of symbols in the constellation
        code_label: str, the type of code ("Gray" or "Binary")
    """

    print(f"\n ------ Modulation type: {modulation_type}, M: {M}, Code label: {code_label} ------")

    # Calculate N0 for a given Eb/N0 in dB
    Eb = 1
    EbN0_dB = 6
    EbN0_linear = 10**(EbN0_dB/10)

    N_0 = Eb / EbN0_linear
    print(f"\nEb/N0 = {EbN0_dB} dB, which corresponds to N0 = {N_0:.3f}")
    
    # Modulation
    mod_symbols, mod_symbol_idxs = modulate_symbols(binary_vector, modulation_type, M, code_label)
    print(f"\nConstellation points for {modulation_type} modulation with M={M} and {code_label} code:", mod_symbols, sep="\n")

    # Plot transmitted constellation
    plot_constellation(modulation_type, mod_symbols, M, MEDIA_CHANNEL_PATH / f"{EbN0_dB}dB", code_label=code_label, filename=f"{modulation_type}_{M}_{code_label}_constellation_transmitted.png")

    # Channel effects
    channel_symbols = channel_effects(mod_symbols, N_0)    
     
    # Plot received constellation
    plot_constellation(modulation_type, channel_symbols, M, MEDIA_CHANNEL_PATH / f"{EbN0_dB}dB", code_label=code_label, filename=f"{modulation_type}_{M}_{code_label}_constellation_received.png")

    # Demodulation without channel effects & error checking
    demod_symbols, demod_symbol_idxs = demodulate_symbols(mod_symbols, modulation_type, M, code_label, original_length=len(binary_vector))
    if np.array_equal(mod_symbol_idxs, demod_symbol_idxs) and np.array_equal(binary_vector, demod_symbols):
        print(f"\nDemodulation without channel effects successful: All symbols correctly demodulated.")
    else:
        num_errors = np.sum(mod_symbol_idxs != demod_symbol_idxs)
        total_symbols = len(mod_symbol_idxs)
        print(f"\nDemodulation without channel effects had errors: {num_errors} out of {total_symbols} symbols were incorrectly demodulated.")

    # Demodulation with channel effects & error checking
    demod_symbols, demod_symbol_idxs = demodulate_symbols(channel_symbols, modulation_type, M, code_label, original_length=len(binary_vector))
    if np.array_equal(mod_symbol_idxs, demod_symbol_idxs) and np.array_equal(binary_vector, demod_symbols):
        print(f"\nDemodulation with channel effects successful: All symbols correctly demodulated.")
    else:
        num_errors = np.sum(mod_symbol_idxs != demod_symbol_idxs)
        total_symbols = len(mod_symbol_idxs)
        print(f"\nDemodulation with channel effects had errors: {num_errors} out of {total_symbols} symbols were incorrectly demodulated.")

if __name__ == "__main__":
    with open(TXT_PATH, 'r') as f:
        text = f.read()

    # Source codification
    probs_dict, char_counts_dict = appearence_probs(text)
    code_dict = huffman_algorithm(probs_dict)
    codified_text = codificate_text(text, code_dict) # ['10011100', '1111111', ... ] for each char in the text 
  
    # Adapt the output of codification to be a binary vector (list of 0s and 1s)
    binary_vector = np.array([int(bit) for symbol in codified_text for bit in symbol]) # [1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, ... ]

    # Test channel effects for both modulation types
    channel(binary_vector, modulation_type="QAM", M=16, code_label="Gray")
    channel(binary_vector, modulation_type="FSK", M=2, code_label="Binary")    
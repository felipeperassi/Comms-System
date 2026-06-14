import numpy as np
from data.config import PATH, OUTPUT_PATH, MEDIA_PATH
from scripts.transmitter import (
    appearence_probs, entropy, huffman_algorithm, mean_length, minimum_length, shannon_range, codificate_text, 
    modulate_symbols, calculate_energies,codificate_channel
)
from scripts.receiver import decode_text, write_file, code_parameters, decodificate_channel, syndrome_table, parity, demodulate_symbols, symbol_error_probability, bit_error_probability
from scripts.extras import plot_char_counts, print_dict, plot_constellation
from scripts.channel import channel_effects


probs_dict = {
    "A": 0.4,
    "B": 0.3,
    "C": 0.2,
    "D": 0.1
}

code_dict = huffman_algorithm(probs_dict)
print_dict(code_dict, "Huffman codes:", sort=True)
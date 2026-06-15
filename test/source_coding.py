from data.config import MEDIA_PATH, TXT_PATH
from scripts.transmitter import (
    appearence_probs, entropy, huffman_algorithm, mean_length, 
    minimum_length, shannon_range, codificate_text
)
from scripts.receiver import decode_text, write_file
from scripts.extras import plot_char_counts, print_dict

MEDIA_CODING_PATH = MEDIA_PATH / "coding"

def source_coding(text : str) -> None:
    """
    Performs source encoding & decoding using Huffman coding for a given text.
    
    Parameters:
        text: str, the input text to be encoded and decoded
    """

    # Calculate probabilities & counts
    probs_dict, char_counts_dict = appearence_probs(text)
    print_dict(char_counts_dict, "Character counts:", sort=True)
    print_dict(probs_dict, "Appearance probabilities:", sort=True)
    plot_char_counts(char_counts_dict, MEDIA_CODING_PATH)
    
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
           f"Minimum code length: {min_len:.3f} bits/symbol", sep="\n")

    # Codification and decoding
    codified_text = codificate_text(text, code_dict)
    decoded_text = decode_text(codified_text, code_dict)
    write_file(MEDIA_CODING_PATH / "decoded_output.txt", decoded_text)

    # Sentence to test the codification and decoding
    text_sentence = text.split('.')[1][1::] + '.' # take the first sentence of the text, removing the leading space and adding the dot at the end
    codified_sentence = codificate_text(text_sentence, code_dict)
    decoded_sentence = decode_text(codified_sentence, code_dict)
    print(f"\nOriginal sentence: {text_sentence}", 
          f"Codified sentence: {' '.join(codified_sentence)}", 
          f"Decoded sentence: {''.join(decoded_sentence)}", sep="\n")

if __name__ == "__main__":
    with open(TXT_PATH, 'r',encoding='utf-8') as f:
        text = f.read()
    
    source_coding(text)
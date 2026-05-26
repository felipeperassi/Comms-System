from data.config import PATH, OUTPUT_PATH
from scripts.transmitter import appearence_probs, entropy, huffman_algorithm, mean_length, minimum_length, shannon_range, codificate_text
from scripts.receiver import decode_text, write_file

def print_dict(dict, title=None, sort=False) -> None:
    """
    prints a dictionary in a readable format, optionally with a title and sorted by values.
    Parameters:
        dict: the dictionary to be printed
        title: optional string to be printed as a title before the dictionary
        sort: if True, sorts the dictionary by values in descending order (and by keys in case of ties)
    """
    if title:
        print(f"\n{title}")
        print("-" * 30)
    items = dict.items()
    if sort:
        try:
            items = sorted(items, key=lambda x: (-x[1], x[0]))
        except Exception:
            items = sorted(items, key=lambda x: x[0])
    for key, value in items:
        print(f"  {repr(key)}: {value}")
    print("-" * 30 + "\n")

def tp1():
    with open(PATH, 'r') as f:
        text = f.read()
    
    # Calculate probabilities & counts
    probs_dict, char_counts_dict = appearence_probs(text)
    print_dict(char_counts_dict, "Character counts:", sort=True)
    print_dict(probs_dict, "Appearance probabilities:", sort=True)
    
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
    codified_codes = codificate_text(text, code_dict)

    decoded_symbols = decode_text(codified_codes, code_dict)
    write_file(OUTPUT_PATH + "decoded_output.txt", decoded_symbols)

    # Sentence to test the codification and decoding
    text_sentence = text.split('.')[1] + '.' # take the first sentence of the text
    codified_sentence = codificate_text(text_sentence, code_dict)
    decoded_sentence = decode_text(codified_sentence, code_dict)
    print(f"\nOriginal sentence: {text_sentence}", 
          f"Codified sentence: {' '.join(codified_sentence)}", 
          f"Decoded sentence: {''.join(decoded_sentence)}", sep="\n")

if __name__ == "__main__":
    tp1()
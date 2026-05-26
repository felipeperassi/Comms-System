from data.config import PATH, OUTPUT_PATH
from scripts.transmitter import appearence_probs, entropy, huffman_algorithm, mean_length, minimum_length, shannon_range, codificate_text
from scripts.receiver import decode_text, write_file

def main():
    with open(PATH, 'r') as f:
        text = f.read()
    
    probs_dict = appearence_probs(text, False, True)

    entropy_value = entropy(probs_dict)
    print("Entropy:", entropy_value)

    code_dict = huffman_algorithm(probs_dict)
    print("Huffman codes:", code_dict)

    mean_len = mean_length(code_dict, probs_dict)
    print("Mean code length:", mean_len)

    min_len = minimum_length(code_dict)
    print("Minimum code length:", min_len)

    shannon_rng = shannon_range(entropy_value)
    print("Shannon range:", shannon_rng)

    codified_codes = codificate_text(text, code_dict)
    decoded_symbols = decode_text(codified_codes, code_dict)

    write_file(OUTPUT_PATH, decoded_symbols)

if __name__ == "__main__":
    main()
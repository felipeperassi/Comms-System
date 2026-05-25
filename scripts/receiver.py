
def decode_text(codified_codes, code_dict) -> list:
    """
    Decodes a list of codified binary codes using the provided code dictionary.
    
    Parameters:
        codified_codes: list of str, the codified binary codes to be decoded
        code_dict: dict {symbol: code}
    
    Returns:
        list: the decoded symbols corresponding to the codified codes
    """
    reverse_code_dict = {code: symbol for symbol, code in code_dict.items()} # reverse mapping from code to symbol
    return [reverse_code_dict[code] for code in codified_codes]

def write_file(filename, decoded_list) -> None:
    """
    Writes a list of decoded symbols to a file.

    Parameters:
        filename: str, the name of the file to write to
        decoded_list: list, the list of decoded symbols to be written to the file
    """
    content = ''.join(decoded_list)
    with open(filename, 'w') as f:
        f.write(content)
import os
import matplotlib.pyplot as plt

def plot_char_counts(char_counts: dict, output_dir: str, filename: str = "char_counts.png") -> None:
    """
    Plots a horizontal bar chart of character counts and saves it as an image file.

    Parameters:
        char_counts: dict {character: count}
        output_dir: str, the directory where the image will be saved
        filename: str, the name of the image file (default: "char_counts.png")
    """
    sorted_items = sorted(char_counts.items(), key=lambda x: x[1], reverse=True)
    labels = [repr(k) for k, _ in sorted_items]
    values = [v for _, v in sorted_items]

    fig, ax = plt.subplots(figsize=(10, 14))
    ax.barh(labels, values, color="steelblue")
    ax.invert_yaxis()
    ax.set_xlabel("Cantidad de apariciones")
    ax.set_title("Frecuencia de caracteres")
    ax.tick_params(axis="y", labelsize=8)

    for i, v in enumerate(values):
        ax.text(v + 2, i, str(v), va="center", fontsize=7)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, filename), dpi=150)
    plt.close()

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
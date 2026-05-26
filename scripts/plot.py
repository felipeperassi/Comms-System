import os
import matplotlib.pyplot as plt


def plot_char_counts(char_counts: dict, output_dir: str, filename: str = "char_counts.png") -> None:
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

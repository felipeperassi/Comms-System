import os
import numpy as np
import matplotlib.pyplot as plt
from data.config import MEDIA_PATH
from scripts.transmitter import amplitude_label

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

def qam_reference(M: int, code_label: str) -> tuple:
    """
    Reconstructs the full normalized M-QAM reference constellation together with
    the bit label of each point, matching the transmitter's grid in
    modulate_symbols. The label is the k-bit word that maps to that point; the
    code_label ("Binary" or "Gray") only affects the point positions.

    Parameters:
        M: int, the number of symbols in the constellation
        code_label: str, the type of code ("Gray" or "Binary")
    Returns:
        tuple: (points, labels) where points is an (M, 2) array of energy-
               normalized I/Q coordinates and labels is a list of k-bit strings
    """
    k = int(np.log2(M))
    kI, kQ = int(np.ceil(k / 2)), int(np.floor(k / 2))
    MI, MQ = 2 ** kI, 2 ** kQ

    points = np.zeros((M, 2))
    labels = []
    for number in range(M):
        bits = np.array(list(np.binary_repr(number, width=k)), dtype=int)
        coordI = amplitude_label(bits[:kI], MI, code_label)
        coordQ = amplitude_label(bits[kI:], MQ, code_label)
        points[number] = [coordI, coordQ]
        labels.append(np.binary_repr(number, width=k))

    Es = k  # Eb = 1, so Es = Eb * k = k
    Es_grid = np.mean(np.sum(points ** 2, axis=1))
    points *= np.sqrt(Es / Es_grid)

    return points, labels

def plot_constellation(modulation_type : str, constellation: np.array, M: int, output_dir: str, code_label: str = "Binary", filename: str = "constellation.png") -> None:
    """
    Plots the constellation points and saves it as an image file. For QAM it also
draws the minimum-distance decision regions and the bit labels of each symbol.

    Parameters:
        modulation_type: str, the type of modulation
        constellation: np.array of shape (N, 2) representing the constellation points
        M: int, the number of symbols in the modulation scheme (e.g., M=16 for 16-QAM)
        output_dir: str, the directory where the image will be saved
        code_label: str, the type of code ("Gray" or "Binary"), used for QAM labels
        filename: str, the name of the image file (default: "constellation.png")
    """
    supported_modulations = ["QAM", "FSK"]
    if modulation_type not in supported_modulations:
        raise ValueError(f"Unsupported modulation type: {modulation_type}. Supported types are: {supported_modulations}")

    if modulation_type == "FSK" and M != 2:
        raise ValueError(f"FSK modulation only supports M=2. Received M={M}.")

    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    ax.set_axisbelow(True)

    ax.grid(alpha=0.3)
    ax.scatter(constellation[:, 0], constellation[:, 1], marker="x")
    if modulation_type == "QAM":
        ax.set_xlabel("In-phase")
        ax.set_ylabel("Quadrature")

        # Reference constellation (receiver's decision grid) with bit labels.
        ref_points, ref_labels = qam_reference(M, code_label)

        # Decision regions: midpoints between adjacent amplitude levels. The QAM
        # grid is separable in I/Q, so the optimum (minimum-distance) regions are
        # bounded by vertical and horizontal lines.
        levels_I = np.unique(np.round(ref_points[:, 0], 9))
        levels_Q = np.unique(np.round(ref_points[:, 1], 9))
        bounds_I = (levels_I[:-1] + levels_I[1:]) / 2
        bounds_Q = (levels_Q[:-1] + levels_Q[1:]) / 2
        for i, x in enumerate(bounds_I):
            ax.axvline(x, color="red", linestyle="--", linewidth=1, alpha=0.6,
                       zorder=1, label="Decision region" if i == 0 else None)
        for y in bounds_Q:
            ax.axhline(y, color="red", linestyle="--", linewidth=1, alpha=0.6, zorder=1)

        # Symbol labeling: annotate each reference point with its k-bit word.
        for (x, y), label in zip(ref_points, ref_labels):
            ax.annotate(label, (x, y), textcoords="offset points", xytext=(5, 5),
                        fontsize=7, color="darkgreen", zorder=3)

        if len(bounds_I) or len(bounds_Q):
            ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    elif modulation_type == "FSK":
        ax.set_xlabel(r"$\psi_1(t) = \sqrt{\frac{2}{T_b}} ~ \cos(\omega_1 t)$")
        ax.set_ylabel(r"$\psi_2(t) = \sqrt{\frac{2}{T_b}} ~ \cos(\omega_2 t)$")
    ax.set_title(f"Constellation Diagram - {modulation_type}")
    ax.axis('equal')

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches="tight")
    plt.close()

def plot_error_curve(EbN0_vec, theo_error, sim_error_cod, sim_error_uncod, title="", y_label=r"$P$", filename="error.png") -> None:
    """
    Plots theoretical and simulated error probability series as a function of Eb/N0

    Parameters:
        EbN0_vec: array-like, the Eb/N0 values in dB (x axis)
        theo_error: array-like, the theoretical error probabilities
        sim_error_cod: array-like, the simulated error probabilities for the coded case
        sim_error_uncod: array-like, the simulated error probabilities for the uncoded case
        title: str, the plot title
        y_label: str, the y axis label (e.g. r"$P_e$" or r"$P_b$")
        filename: str, the output image file name
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.semilogy(EbN0_vec, theo_error, "-", label="Theoretical")
    ax.semilogy(EbN0_vec, sim_error_uncod, "x--", label="Simulated (Uncoded)")
    ax.semilogy(EbN0_vec, sim_error_cod, "x--", label="Simulated (Coded)")
    
    ax.set(xlabel=r"$E_b/N_0$ [dB]", ylabel=y_label, title=title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    os.makedirs(MEDIA_PATH / "error_curves", exist_ok=True)
    fig.savefig(MEDIA_PATH / "error_curves" / filename, dpi=150)
    plt.close(fig)

def plot_error_curves(EbN0_dB, series, filename, suptitle="") -> None:
    """
    Plots a single figure with two subplots, Pe (left) and Pb (right), as a
    function of Eb/N0. Each entry in ``series`` produces one simulated curve
    (markers) and, optionally, one theoretical curve (solid line) in the same
    color on both subplots.

    Parameters:
        EbN0_dB: array-like, the Eb/N0 values in dB (shared x axis)
        series: list of dicts, each with keys:
            "label": str, the curve label (e.g. "M=2")
            "pe_sim", "pb_sim": array-like, simulated symbol/bit error probs
            "pe_theo", "pb_theo": array-like or None, theoretical symbol/bit probs
        filename: str, the output image file name
        suptitle: str, the overall figure title
    """
    fig, (ax_pe, ax_pb) = plt.subplots(1, 2, figsize=(13, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for i, s in enumerate(series):
        c = colors[i % 10]
        label = s["label"]

        if s.get("pe_theo") is not None:
            ax_pe.semilogy(EbN0_dB, s["pe_theo"], "-", color=c, label=f"{label} (teórica)")
        ax_pe.semilogy(EbN0_dB, s["pe_sim"], "x", color=c, label=f"{label} (simulada)")

        if s.get("pb_theo") is not None:
            ax_pb.semilogy(EbN0_dB, s["pb_theo"], "-", color=c, label=f"{label} (teórica)")
        ax_pb.semilogy(EbN0_dB, s["pb_sim"], "x", color=c, label=f"{label} (simulada)")

    ax_pe.set(xlabel=r"$E_b/N_0$ [dB]", ylabel=r"$P_e$", title="Probabilidad de error de símbolo")
    ax_pb.set(xlabel=r"$E_b/N_0$ [dB]", ylabel=r"$P_b$", title="Probabilidad de error de bit")
    for ax in (ax_pe, ax_pb):
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    if suptitle:
        fig.suptitle(suptitle)

    fig.tight_layout()
    os.makedirs(MEDIA_PATH / "error_curves", exist_ok=True)
    fig.savefig(MEDIA_PATH / "error_curves" / filename, dpi=150)
    plt.close(fig)

def plot_modulation_comparison(EbN0_dB, panels, filename, suptitle="") -> None:
    """
    Plots one subplot per M value, each comparing the theoretical bit error
    probability of several modulations as a function of Eb/N0.

    Parameters:
        EbN0_dB: array-like, the Eb/N0 values in dB (shared x axis)
        panels: list of (M, curves) tuples, where curves is a dict
                {modulation_name: Pb_array} for that M
        filename: str, the output image file name
        suptitle: str, the overall figure title
    """
    ncols = 2
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 5 * nrows), squeeze=False)

    for ax, (M, curves) in zip(axes.flat, panels):
        for name, pb in curves.items():
            ax.semilogy(EbN0_dB, pb, label=name)
        ax.set(xlabel=r"$E_b/N_0$ [dB]", ylabel=r"$P_b$", title=f"M = {M}")
        ax.set_ylim(1e-6, 1)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)

    for ax in axes.flat[len(panels):]:
        ax.axis("off")
    if suptitle:
        fig.suptitle(suptitle)

    fig.tight_layout()
    os.makedirs(MEDIA_PATH / "error_curves", exist_ok=True)
    fig.savefig(MEDIA_PATH / "error_curves" / filename, dpi=150)
    plt.close(fig)

def plot_simulated_comparison(EbN0_dB, curves, y_label, title, filename) -> None:
    """
    Plots several simulated error-probability series on a single semilog axis,
    e.g. to compare the uncoded and channel-coded cases.

    Parameters:
        EbN0_dB: array-like, the Eb/N0 values in dB (x axis)
        curves: list of dicts, each with keys "label" and "values"
        y_label: str, the y axis label (e.g. r"$P_e$" or r"$P_b$")
        title: str, the plot title
        filename: str, the output image file name
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    for curve in curves:
        ax.semilogy(EbN0_dB, curve["values"], "x-", label=curve["label"])

    ax.set(xlabel=r"$E_b/N_0$ [dB]", ylabel=y_label, title=title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    os.makedirs(MEDIA_PATH / "error_curves", exist_ok=True)
    fig.savefig(MEDIA_PATH / "error_curves" / filename, dpi=150)
    plt.close(fig)
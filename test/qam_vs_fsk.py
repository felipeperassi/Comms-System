"""
Comparacion QAM vs FSK al crecer M, simulada sobre el texto real
(relativity_einstein.txt).

Muestra el compromiso clasico entre modulaciones eficientes en ancho de banda
y eficientes en energia:

  * QAM (eficiente en ancho de banda): al crecer M entran mas bits por simbolo
    en el mismo ancho de banda, pero los puntos se acercan y la Pb EMPEORA
    (las curvas se corren a la derecha, hace falta mas Eb/N0).

  * FSK (eficiente en energia): al crecer M las senales siguen siendo
    ortogonales, la Pb MEJORA (las curvas se corren a la izquierda), pero a
    costa de un ancho de banda que crece con M (M tonos ortogonales).

Uso:  python -m test.qam_vs_fsk
"""
import numpy as np
import matplotlib.pyplot as plt

from data.config import TXT_PATH, MEDIA_PATH
from scripts.transmitter import (appearence_probs, huffman_algorithm,
                                 codificate_text, modulate_symbols)
from scripts.receiver import (demodulate_symbols, symbol_error_probability,
                              bit_error_probability)
from scripts.channel import channel_effects
from test.error_curves import (qam_bit_error, fsk_bit_error,
                               qam_symbol_error, fsk_symbol_error, _mask_zeros)

# --------------------------------------------------------------------------- #
#  Parametros
# --------------------------------------------------------------------------- #
M_VEC = [2, 4, 8, 16]
EbN0_dB = np.arange(0, 11)          # 0 .. 10 dB
QAM_LABEL = "Gray"                  # Gray coincide con la Pb teorica
FSK_LABEL = "Binary"               # Gray no aplica a FSK
TARGET_PB = 1e-4                   # Pb objetivo para la tabla comparativa
OUT_DIR = MEDIA_PATH / "error_curves"


def load_text_bits():
    """Texto -> Huffman -> vector binario (misma cadena que transmite el sistema)."""
    with open(TXT_PATH, 'r') as f:
        text = f.read()
    probs_dict, _ = appearence_probs(text)
    code_dict = huffman_algorithm(probs_dict)
    codified_text = codificate_text(text, code_dict)
    return np.array([int(bit) for symbol in codified_text for bit in symbol])


def simulate(modulation_type, M, code_label, binary_vector):
    """Barrido de Eb/N0 sobre el texto: devuelve (Pe, Pb) con ceros enmascarados."""
    symbols, tx_idx = modulate_symbols(binary_vector, modulation_type, M, code_label)
    pe_list, pb_list = [], []
    for dB in EbN0_dB:
        N0 = 1.0 / 10 ** (dB / 10)          # Eb = 1  ->  N0 = 1 / (Eb/N0)
        rx = channel_effects(symbols, N0, attenuation=False)
        bits_hat, rx_idx = demodulate_symbols(rx, modulation_type, M, code_label,
                                              original_length=len(binary_vector))
        pe = symbol_error_probability(tx_idx, rx_idx)
        pb = bit_error_probability(binary_vector, bits_hat)
        pe_list.append(pe)
        pb_list.append(pb)
        print(f"  {modulation_type} M={M:2d}  Eb/N0={dB:2d} dB  ->  Pe={pe:.2e}  Pb={pb:.2e}")
    return _mask_zeros(pe_list), _mask_zeros(pb_list)


def ebn0_for_target(bit_error_fn, M, target=TARGET_PB):
    """Eb/N0 [dB] teorico necesario para alcanzar una Pb objetivo (interpolado)."""
    grid = np.linspace(0, 30, 6001)
    pb = np.asarray(bit_error_fn(M, 10 ** (grid / 10)), float)
    below = np.where(pb <= target)[0]
    return grid[below[0]] if below.size else np.nan


def plot_comparison(qam_pb, fsk_pb):
    """Dos paneles (QAM | FSK) con la Pb simulada y teorica de cada M."""
    EbN0_lin = 10 ** (EbN0_dB / 10)
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(M_VEC)))

    fig, (axq, axf) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for c, M in zip(colors, M_VEC):
        axq.semilogy(EbN0_dB, qam_pb[M], "x--", color=c, label=f"M={M} (sim)")
        axq.semilogy(EbN0_dB, qam_bit_error(M, EbN0_lin), "-", color=c, alpha=0.5)
        axf.semilogy(EbN0_dB, fsk_pb[M], "x--", color=c, label=f"M={M} (sim)")
        axf.semilogy(EbN0_dB, fsk_bit_error(M, EbN0_lin), "-", color=c, alpha=0.5)

    axq.set_title("QAM: eficiente en ancho de banda\n(Pb empeora al crecer M)")
    axf.set_title("FSK: eficiente en energia\n(Pb mejora al crecer M)")
    for ax in (axq, axf):
        ax.set(xlabel=r"$E_b/N_0$ [dB]", ylabel=r"$P_b$")
        ax.set_ylim(1e-6, 1)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8, title="lineas: teorica")
    fig.suptitle("QAM vs FSK al crecer M  (simulado sobre relativity_einstein.txt)")
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "qam_vs_fsk_pb.png", dpi=150)
    plt.close(fig)
    print(f"\n-> figura guardada en {OUT_DIR / 'qam_vs_fsk_pb.png'}")


def plot_overlay(qam_data, fsk_data, ylabel, title, filename):
    """QAM y FSK superpuestas en un solo eje: QAM en linea llena, FSK punteada,
    M codificado por color. Solo curvas simuladas para no saturar la figura."""
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(M_VEC)))

    fig, ax = plt.subplots(figsize=(9, 6))
    for c, M in zip(colors, M_VEC):
        ax.semilogy(EbN0_dB, qam_data[M], "x-", color=c, label=f"QAM M={M}")
        ax.semilogy(EbN0_dB, fsk_data[M], "o--", color=c, markersize=4,
                    markerfacecolor="none", label=f"FSK M={M}")

    ax.set(xlabel=r"$E_b/N_0$ [dB]", ylabel=ylabel, title=title)
    ax.set_ylim(1e-6, 1)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / filename, dpi=150)
    plt.close(fig)
    print(f"-> figura guardada en {OUT_DIR / filename}")


def main():
    np.random.seed(0)  # reproducibilidad
    binary_vector = load_text_bits()
    print(f"Bits de fuente (texto): {len(binary_vector):,}\n")

    qam_pe, qam_pb, fsk_pe, fsk_pb = {}, {}, {}, {}
    print("QAM sobre el texto...")
    for M in M_VEC:
        pe, pb = simulate("QAM", M, QAM_LABEL, binary_vector)
        qam_pe[M], qam_pb[M] = pe, pb
    print("\nFSK sobre el texto...")
    for M in M_VEC:
        pe, pb = simulate("FSK", M, FSK_LABEL, binary_vector)
        fsk_pe[M], fsk_pb[M] = pe, pb

    plot_comparison(qam_pb, fsk_pb)
    plot_overlay(qam_pb, fsk_pb, r"$P_b$",
                 "QAM vs FSK: Pb (texto relativity_einstein)",
                 "qam_vs_fsk_pb_overlay.png")
    plot_overlay(qam_pe, fsk_pe, r"$P_e$",
                 "QAM vs FSK: Pe (texto relativity_einstein)",
                 "qam_vs_fsk_pe_overlay.png")

    # Tabla: Eb/N0 [dB] necesario para Pb objetivo, segun crece M (teorico)
    print(f"\nEb/N0 [dB] necesario para Pb = {TARGET_PB:g}  (menor = mas eficiente en energia)")
    print(f"{'M':>3} {'QAM':>8} {'FSK':>8}")
    print("-" * 21)
    for M in M_VEC:
        eq = ebn0_for_target(qam_bit_error, M)
        ef = ebn0_for_target(fsk_bit_error, M)
        print(f"{M:>3} {eq:>8.2f} {ef:>8.2f}")


if __name__ == "__main__":
    main()

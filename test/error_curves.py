import numpy as np
from scipy.special import erfc, comb
from scipy.stats import norm

from scripts.transmitter import modulate_symbols, codificate_channel
from scripts.receiver import (demodulate_symbols, symbol_error_probability,
                              bit_error_probability, parity, syndrome_table,
                              decodificate_channel, code_parameters)
from scripts.channel import channel_effects
from scripts.extras import (plot_error_curves, plot_modulation_comparison,
                            plot_simulated_comparison)

# --------------------------------------------------------------------------- #
#  Theoretical error probabilities                                            #
# --------------------------------------------------------------------------- #

def qfunc(x):
    """Gaussian Q-function, Q(x) = 0.5 * erfc(x / sqrt(2))."""
    return 0.5 * erfc(np.asarray(x, float) / np.sqrt(2))

def qam_symbol_error(M, EbN0_lin):
    """Symbol error probability of square M-QAM (coherent, AWGN)."""
    k = np.log2(M)
    EsN0 = k * np.asarray(EbN0_lin, float)
    a = 2 * (1 - 1 / np.sqrt(M)) * qfunc(np.sqrt(3 * EsN0 / (M - 1)))
    return 1 - (1 - a) ** 2

def qam_bit_error(M, EbN0_lin):
    """Bit error probability of Gray-coded square M-QAM (dominant term)."""
    k = np.log2(M)
    g = np.asarray(EbN0_lin, float)
    return (4 / k) * (1 - 1 / np.sqrt(M)) * qfunc(np.sqrt(3 * k * g / (M - 1)))

def fsk_symbol_error(M, EbN0_lin):
    """Symbol error probability of coherent orthogonal M-FSK (AWGN).

    Ps = 1 - integral_{-inf}^{inf} phi(t) * Phi(t + sqrt(2*Es/N0))^(M-1) dt
    """
    g = np.atleast_1d(np.asarray(EbN0_lin, float))
    EsN0 = np.log2(M) * g
    t = np.linspace(-15, 15, 6001)
    phi = norm.pdf(t)
    Ps = np.empty_like(EsN0)
    for i, gs in enumerate(EsN0):
        Ps[i] = 1 - np.trapz(phi * norm.cdf(t + np.sqrt(2 * gs)) ** (M - 1), t)
    return Ps

def fsk_bit_error(M, EbN0_lin):
    """Bit error probability of orthogonal M-FSK: Pb = Ps * (M/2)/(M-1)."""
    return fsk_symbol_error(M, EbN0_lin) * (M / 2) / (M - 1)

def pam_bit_error(M, EbN0_lin):
    """Bit error probability of Gray-coded bipolar M-PAM (AWGN)."""
    k = np.log2(M)
    g = np.asarray(EbN0_lin, float)
    Ps = 2 * (1 - 1 / M) * qfunc(np.sqrt(6 * k * g / (M ** 2 - 1)))
    return Ps / k

def ask_bit_error(M, EbN0_lin):
    """Bit error probability of Gray-coded unipolar M-ASK (AWGN)."""
    k = np.log2(M)
    g = np.asarray(EbN0_lin, float)
    Ps = 2 * (1 - 1 / M) * qfunc(np.sqrt(3 * k * g / ((M - 1) * (2 * M - 1))))
    return Ps / k

def psk_bit_error(M, EbN0_lin):
    """Bit error probability of Gray-coded M-PSK (AWGN)."""
    k = np.log2(M)
    g = np.asarray(EbN0_lin, float)
    if M == 2:
        Ps = qfunc(np.sqrt(2 * g))                       # BPSK (exact)
    else:
        Ps = 2 * qfunc(np.sqrt(2 * k * g) * np.sin(np.pi / M))
    return Ps / k

def coded_bit_error_bound(p, n, k, t):
    """Info-bit error bound of an (n, k) block code that corrects t errors,
    over a BSC with crossover probability p (the modulation bit error)."""
    p = np.asarray(p, float)
    Pb = np.zeros_like(p)
    for i in range(t + 1, n + 1):
        Pb += i * comb(n, i) * p ** i * (1 - p) ** (n - i)
    return Pb / n

# Dispatch tables so the sweep code is modulation-agnostic.
SYMBOL_ERROR = {"QAM": qam_symbol_error, "FSK": fsk_symbol_error}
BIT_ERROR = {"QAM": qam_bit_error, "FSK": fsk_bit_error}

# --------------------------------------------------------------------------- #
#  Monte Carlo simulation                                                     #
# --------------------------------------------------------------------------- #

def simulate_point(modulation_type, M, code_label, EbN0_dB, n_bits, code=None):
    """Simulates one Eb/N0 operating point and returns (Pe, Pb).

    Without coding (code=None): n_bits information bits are modulated and sent
    over a pure-AWGN channel. With coding (code=(G, k, n, H, S, t)): the bits
    are first block-encoded; the noise is scaled by the code rate R = k/n so the
    x axis remains energy per *information* bit (Eb/N0).
    """
    EbN0_lin = 10 ** (EbN0_dB / 10)

    if code is None:
        bits = np.random.randint(0, 2, n_bits)
        symbols, tx_idx = modulate_symbols(bits, modulation_type, M, code_label)
        N0 = 1 / EbN0_lin
        rx = channel_effects(symbols, N0, attenuation=False)
        bits_hat, rx_idx = demodulate_symbols(rx, modulation_type, M, code_label,
                                              original_length=len(bits))
        return symbol_error_probability(tx_idx, rx_idx), bit_error_probability(bits, bits_hat)

    G, k, n, H, S, t = code
    R = k / n
    info = np.random.randint(0, 2, n_bits)
    coded = codificate_channel(info, G, k, n)
    symbols, tx_idx = modulate_symbols(coded, modulation_type, M, code_label)
    N0 = 1 / (R * EbN0_lin)
    rx = channel_effects(symbols, N0, attenuation=False)
    coded_hat, rx_idx = demodulate_symbols(rx, modulation_type, M, code_label,
                                           original_length=len(coded))
    info_hat = decodificate_channel(coded_hat, H, S, k, n)
    return symbol_error_probability(tx_idx, rx_idx), bit_error_probability(info, info_hat)

def _mask_zeros(values):
    """Replaces 0 with NaN so the log-scale plot skips error-free points."""
    arr = np.array(values, float)
    arr[arr == 0] = np.nan
    return arr

def build_series(modulation_type, M_list, code_label_of, EbN0_dB, n_bits, code=None):
    """Runs the sweep over Eb/N0 for every M and returns the plot series list."""
    EbN0_lin = 10 ** (EbN0_dB / 10)
    R = code[1] / code[2] if code is not None else 1.0
    series = []
    for M in M_list:
        code_label = code_label_of(M)
        pe_sim, pb_sim = [], []
        for dB in EbN0_dB:
            pe, pb = simulate_point(modulation_type, M, code_label, dB, n_bits, code)
            pe_sim.append(pe)
            pb_sim.append(pb)
            print(f"  {modulation_type} M={M:2d} {'coded' if code else 'uncoded':7s} "
                  f"Eb/N0={dB:2d} dB -> Pe={pe:.2e} Pb={pb:.2e}")

        if code is None:
            pe_theo = SYMBOL_ERROR[modulation_type](M, EbN0_lin)
            pb_theo = BIT_ERROR[modulation_type](M, EbN0_lin)
        else:
            # Symbols are not corrected: modulation symbol error at the rate-
            # shifted operating point. Bits are corrected: block-code bound.
            gmod = R * EbN0_lin
            pe_theo = SYMBOL_ERROR[modulation_type](M, gmod)
            p_chan = BIT_ERROR[modulation_type](M, gmod)
            pb_theo = coded_bit_error_bound(p_chan, code[2], code[1], code[5])

        series.append({"label": f"M={M}",
                       "pe_sim": _mask_zeros(pe_sim), "pb_sim": _mask_zeros(pb_sim),
                       "pe_theo": pe_theo, "pb_theo": pb_theo})
    return series

# --------------------------------------------------------------------------- #
#  Main: four figures (FSK / QAM, with / without channel coding)              #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    np.random.seed(0)  # reproducible figures

    EbN0_dB = np.arange(0, 11)          # 0 .. 10 dB, 1 dB steps
    N_UNCODED = 500_000
    N_CODED = 200_000                    # multiple of k = 4

    FSK_M = [2, 4, 8, 16]
    QAM_M = [16]
    fsk_label = lambda M: "Binary"       # Gray not defined for FSK
    qam_label = lambda M: "Gray"         # Gray matches the theoretical BER curve

    # (8, 4) linear block code given by the lecturers
    G = np.array([
        [1, 1, 1, 0, 1, 0, 0, 0],
        [1, 1, 0, 1, 0, 1, 0, 0],
        [1, 0, 1, 1, 0, 0, 1, 0],
        [0, 1, 1, 1, 0, 0, 0, 1],
    ], dtype=int)
    k, n = 4, 8
    H = parity(G, k, n)
    S = syndrome_table(H, n)
    _, _, t = code_parameters(G, k, n)
    code = (G, k, n, H, S, t)

    print("FSK without channel coding...")
    fsk_unc = build_series("FSK", FSK_M, fsk_label, EbN0_dB, N_UNCODED)
    plot_error_curves(EbN0_dB, fsk_unc, "fsk_uncoded.png",
                      suptitle="M-FSK sin codificación de canal")

    print("FSK with channel coding...")
    fsk_cod = build_series("FSK", FSK_M, fsk_label, EbN0_dB, N_CODED, code)
    plot_error_curves(EbN0_dB, fsk_cod, "fsk_coded.png",
                      suptitle="M-FSK con codificación de canal (8,4)")

    print("QAM without channel coding...")
    qam_unc = build_series("QAM", QAM_M, qam_label, EbN0_dB, N_UNCODED)
    plot_error_curves(EbN0_dB, qam_unc, "qam_uncoded.png",
                      suptitle="16-QAM sin codificación de canal")

    print("QAM with channel coding...")
    qam_cod = build_series("QAM", QAM_M, qam_label, EbN0_dB, N_CODED, code)
    plot_error_curves(EbN0_dB, qam_cod, "qam_coded.png",
                      suptitle="16-QAM con codificación de canal (8,4)")

    # ----- Point c: bandwidth-efficient (PAM/ASK/PSK/QAM) vs energy-efficient
    # (FSK) modulations as M grows (theoretical Pb). QAM is only included for
    # square constellations (M = 4, 16). -----
    print("Modulation comparison (point c)...")
    EbN0_lin = 10 ** (EbN0_dB / 10)
    panels = []
    for M in [2, 4, 8, 16]:
        curves = {
            "PAM": pam_bit_error(M, EbN0_lin),
            "ASK": ask_bit_error(M, EbN0_lin),
            "PSK": psk_bit_error(M, EbN0_lin),
            "FSK": fsk_bit_error(M, EbN0_lin),
        }
        if np.sqrt(M) == int(np.sqrt(M)):
            curves["QAM"] = qam_bit_error(M, EbN0_lin)
        panels.append((M, curves))
    plot_modulation_comparison(EbN0_dB, panels, "modulation_comparison.png",
                               suptitle="Comparación de modulaciones (Pb teórica) al crecer M")

    # ----- Points d and e: selected modulation (16-QAM), simulated curves
    # comparing the uncoded case with the (8,4) channel code. -----
    print("Coding comparison for 16-QAM (points d, e)...")
    plot_simulated_comparison(
        EbN0_dB,
        [{"label": "Sin codificación", "values": qam_unc[0]["pe_sim"]},
         {"label": "Con código (8,4)", "values": qam_cod[0]["pe_sim"]}],
        r"$P_e$", "16-QAM: Pe simulada, sin vs con codificación", "qam_pe_coding_comparison.png")
    plot_simulated_comparison(
        EbN0_dB,
        [{"label": "Sin codificación", "values": qam_unc[0]["pb_sim"]},
         {"label": "Con código (8,4)", "values": qam_cod[0]["pb_sim"]}],
        r"$P_b$", "16-QAM: Pb simulada, sin vs con codificación", "qam_pb_coding_comparison.png")

    print("\nDone. Figures saved to media/error_curves/")

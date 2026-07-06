import numpy as np

# ----------------------------------- TP3 -----------------------------------

def awgn(shape, N_0) -> np.ndarray:
    """
    Generates Additive White Gaussian Noise (AWGN) for a given number of samples and noise power spectral density.

    Parameters:
        shape: tuple, the shape of the noise samples to generate (e.g., (N, 2) for QAM or (N, M) for FSK)
        N_0: float, the noise power spectral density
    
    Returns:
        noise: np.ndarray of size shape, the generated AWGN noise samples
    """
    var = np.sqrt(N_0 / 2)
    mean = 0

    return np.random.normal(mean, var, size=shape)

def channel_attenuation(size=None) -> np.ndarray:
    """
    Generates a random channel attenuation factor with uniform distribution
    between 0.5 and 0.9.

    Parameters:
        size: None for a single scalar factor, or an int to get one independent
              factor per symbol (returns an array of that length).

    Returns:
        float or np.ndarray: the attenuation factor(s) in [0.5, 0.9]
    """
    return np.random.uniform(0.5, 0.9, size=size)

def channel_effects(mod_symbols, N_0, attenuation=True) -> np.ndarray:
    """
    Simulates the effects of the channel on the transmitted symbols by applying a
    random attenuation (one factor per symbol) and adding AWGN noise.

    Parameters:
        mod_symbols: np.ndarray, the transmitted modulation symbols (shape (N, 2) for QAM or (N, M) for FSK)
        N_0: float, the noise power spectral density
        attenuation: bool, if True applies a random channel attenuation (section D);
                     set to False for a pure AWGN channel (error-probability curves)

    Returns:
        np.ndarray: the received symbols after channel effects
    """
    mod_symbols = np.asarray(mod_symbols, dtype=float)
    signal = mod_symbols
    if attenuation:
        a = channel_attenuation(size=mod_symbols.shape[0])  # una atenuacion por simbolo
        signal = a[:, None] * mod_symbols
    return signal + awgn(mod_symbols.shape, N_0)
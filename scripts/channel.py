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

def channel_attenuation() -> float:
    """
    Simulates channel attenuation by generating a random attenuation factor between 0.5 and 0.9.
    
    Returns:
        float: the attenuation factor
    """
    return np.random.uniform(0.5, 0.9)

def channel_effects (mod_symbols, N_0) -> np.ndarray:
    """
    Simulates the effects of the channel on the transmitted symbols by applying attenuation and adding AWGN noise.
    
    Parameters:
        mod_symbols: np.ndarray, the transmitted modulation symbols (shape (N, 2) for QAM or (N, M) for FSK)
        N_0: float, the noise power spectral density
    
    Returns:
        np.ndarray: the received symbols after channel effects
    """  
    return mod_symbols * channel_attenuation() + awgn(mod_symbols.shape, N_0)
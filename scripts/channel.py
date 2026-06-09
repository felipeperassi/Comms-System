import numpy as np

def awgn(N, N0):
    """
    Generates Additive White Gaussian Noise (AWGN) for a given number of samples and noise power spectral density.

    Parameters:
        N: int, the number of noise samples to generate
        N0: float, the noise power spectral density
"""
    sigma = np.sqrt(N0 / 2)
    noise_i = np.random.normal(0, sigma, N)
    noise_q = np.random.normal(0, sigma, N)
        
    return np.column_stack([noise_i, noise_q])

def atenuacion():

    #return np.random.uniform(0.5, 0.9)
    return 1

def channel_effects (symbols, N0):

    noise = awgn(len(symbols), N0)

    atenuados = symbols * atenuacion()

    return atenuados + noise


    

"""Channel models and log-likelihood-ratio (LLR) computation.

  BSC(p)  -- binary symmetric channel, bit flips with probability p
  AWGN    -- BPSK modulation (0 -> +1, 1 -> -1) plus Gaussian noise
"""

import numpy as np


def bsc_transmit(codeword: np.ndarray, p: float, rng: np.random.Generator) -> np.ndarray:
    """Flip each bit of `codeword` independently with probability `p`."""
    flips = (rng.random(codeword.shape) < p).astype(np.uint8)
    return codeword ^ flips


def bsc_llr(received: np.ndarray, p: float) -> np.ndarray:
    """LLR for a BSC with crossover probability `p` (0 < p < 0.5).

    LLR(0) = +log((1-p)/p), LLR(1) = -log((1-p)/p).
    """
    l = np.log((1 - p) / p)
    return np.where(received == 0, l, -l)


def bpsk_modulate(codeword: np.ndarray) -> np.ndarray:
    """Map bits to BPSK symbols: 0 -> +1.0, 1 -> -1.0."""
    return 1.0 - 2.0 * codeword.astype(np.float64)


def awgn_transmit(codeword: np.ndarray, snr_db: float, rng: np.random.Generator):
    """Transmit a BPSK-modulated codeword over an AWGN channel at the given
    Eb/N0 (dB), assuming one coded bit per channel use (rate-1 mapping).

    Returns (received_signal, sigma).
    """
    x = bpsk_modulate(codeword)
    snr_linear = 10 ** (snr_db / 10)
    sigma = np.sqrt(1.0 / (2 * snr_linear))
    noise = rng.normal(0.0, sigma, size=x.shape)
    return x + noise, sigma


def awgn_llr(received_signal: np.ndarray, sigma: float) -> np.ndarray:
    """LLR for an AWGN channel with BPSK mapping (0 -> +1, 1 -> -1)."""
    return 2.0 * received_signal / (sigma ** 2)

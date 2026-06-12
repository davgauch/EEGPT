"""
Frequency-domain band-stop masking for EEG signals.
"""

import numpy as np
import torch


BAND_RANGES = {
    "delta":      (1,  4),
    "theta":      (4,  8),
    "alpha":      (8,  12),
    "beta":       (13, 30),
    "beta_upper": (20, 30),
    # Bandwidth grid search (all starting at 4 Hz)
    "theta_bw1":  (4,  5),    # 1 Hz
    "theta_bw2":  (4,  6),    # 2 Hz
    "theta_bw3":  (4,  7),    # 3 Hz
    "theta_bw4":  (4,  8),    # 4 Hz  
    "theta_bw5":  (4,  9),    # 5 Hz
    "theta_bw6":  (4,  10),   # 6 Hz
    "theta_bw8":  (4,  12),   # 8 Hz  
    "theta_bw10": (4,  14),   # 10 Hz
    "theta_bw12": (4,  16),   # 12 Hz 
}


def apply_bandstop_mask(x: np.ndarray, strategy: str, sfreq: int) -> np.ndarray:
    if strategy == "none":
        return x.copy()

    t = torch.tensor(x, dtype=torch.float32).unsqueeze(0)   # (1, C, T)
    spectrum = torch.fft.rfft(t, dim=-1)
    freqs    = torch.fft.rfftfreq(t.shape[-1], d=1.0 / sfreq)

    if strategy == "random":
        bandwidth = 4.0
        center    = np.random.uniform(1.0 + bandwidth / 2, 50.0 - bandwidth / 2)
        low, high = center - bandwidth / 2, center + bandwidth / 2
    elif strategy in BAND_RANGES:
        low, high = BAND_RANGES[strategy]
    else:
        raise ValueError(
            f"Unknown strategy '{strategy}'. "
            f"Choose from: {sorted(BAND_RANGES.keys()) + ['random', 'none']}")

    mask = (freqs >= low) & (freqs < high)
    spectrum[:, :, mask] = 0
    t = torch.fft.irfft(spectrum, n=t.shape[-1], dim=-1)
    return t.squeeze(0).numpy()

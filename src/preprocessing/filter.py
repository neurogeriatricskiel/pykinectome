from scipy.signal import butter, filtfilt
import numpy as np


def butter_lowpass_filter(data: np.ndarray, fs: float, cutoff: float) -> np.ndarray:
    return data

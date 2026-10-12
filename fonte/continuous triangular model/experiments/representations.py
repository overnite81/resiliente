"""ECG representations used in the controlled CTM comparison."""

from typing import Literal

import numpy as np
import pywt
from scipy.signal import stft

from tools.ctm_plot_image_fixed import build_ctm_grid


RepresentationName = Literal["raw", "wavelet", "stft", "ctm"]



def raw_features(signal: np.ndarray) -> np.ndarray:
    """Return the normalized fixed-length ECG window."""
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError("signal deve ser unidimensional.")
    return signal.copy()



def wavelet_features(
    signal: np.ndarray,
    *,
    wavelet: str = "db4",
    level: int | None = None,
) -> np.ndarray:
    """Return concatenated discrete wavelet coefficients."""
    signal = raw_features(signal)
    wavelet_object = pywt.Wavelet(wavelet)
    maximum_level = pywt.dwt_max_level(len(signal), wavelet_object.dec_len)
    if maximum_level < 1:
        raise ValueError("A janela é pequena demais para a transformada Wavelet.")
    selected_level = maximum_level if level is None else level
    if selected_level < 1 or selected_level > maximum_level:
        raise ValueError(f"level deve estar entre 1 e {maximum_level}.")
    coefficients = pywt.wavedec(signal, wavelet_object, level=selected_level)
    return np.concatenate(coefficients).astype(np.float64, copy=False)



def stft_features(
    signal: np.ndarray,
    *,
    nperseg: int = 64,
    noverlap: int = 32,
    log_magnitude: bool = True,
) -> np.ndarray:
    """Return flattened STFT magnitude coefficients."""
    signal = raw_features(signal)
    if nperseg <= 1 or nperseg > len(signal):
        raise ValueError("nperseg deve ser maior que 1 e não exceder a janela.")
    if noverlap < 0 or noverlap >= nperseg:
        raise ValueError("noverlap deve estar entre 0 e nperseg - 1.")

    _, _, coefficients = stft(
        signal,
        nperseg=nperseg,
        noverlap=noverlap,
        boundary=None,
        padded=False,
    )
    magnitude = np.abs(coefficients)
    if log_magnitude:
        magnitude = np.log1p(magnitude)
    return magnitude.ravel().astype(np.float64, copy=False)



def ctm_features(
    signal: np.ndarray,
    *,
    span: int = 32,
    measure: Literal["amplitude", "energy"] = "amplitude",
) -> np.ndarray:
    """Return valid CTM triangular cells in scale-major order.

    ``amplitude`` integrates the interpolated ECG signal. ``energy`` integrates
    squared amplitude and is kept as an explicit experimental variant.
    """
    signal = raw_features(signal)
    if span < 2:
        raise ValueError("span deve ser pelo menos 2.")
    if measure not in {"amplitude", "energy"}:
        raise ValueError("measure deve ser 'amplitude' ou 'energy'.")

    values = signal if measure == "amplitude" else signal**2
    time = np.arange(len(values), dtype=np.float64)
    grid = build_ctm_grid(list(zip(time, values)), span)
    return np.asarray(
        [value for row in grid for value in row if value is not None],
        dtype=np.float64,
    )



def extract_representations(
    signal: np.ndarray,
    *,
    ctm_span: int = 32,
    ctm_measure: Literal["amplitude", "energy"] = "amplitude",
    wavelet: str = "db4",
    wavelet_level: int | None = None,
    stft_nperseg: int = 64,
    stft_noverlap: int = 32,
) -> dict[RepresentationName, np.ndarray]:
    """Extract all comparison representations from one ECG window."""
    return {
        "raw": raw_features(signal),
        "wavelet": wavelet_features(signal, wavelet=wavelet, level=wavelet_level),
        "stft": stft_features(
            signal,
            nperseg=stft_nperseg,
            noverlap=stft_noverlap,
        ),
        "ctm": ctm_features(signal, span=ctm_span, measure=ctm_measure),
    }

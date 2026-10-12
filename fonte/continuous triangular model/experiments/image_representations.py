"""2-D grayscale image encodings for the CPU ResNet pilot."""

from __future__ import annotations

import numpy as np
import pywt
from PIL import Image
from scipy.signal import stft

from tools.ctm_plot_image_fixed import build_ctm_grid


REPRESENTATIONS = ("stft", "wavelet", "ctm")


def _normalize_map(values: np.ndarray, *, valid_mask: np.ndarray | None = None) -> np.ndarray:
    """Robustly map a 2-D representation to [0, 1], preserving CTM padding."""
    values = np.asarray(values, dtype=np.float32)
    mask = np.isfinite(values)
    if valid_mask is not None:
        mask &= valid_mask
    selected = values[mask]
    output = np.zeros_like(values, dtype=np.float32)
    if selected.size == 0:
        return output
    low, high = np.percentile(selected, (1.0, 99.0))
    if high <= low:
        output[mask] = 0.5
        return output
    output[mask] = np.clip((values[mask] - low) / (high - low), 0.0, 1.0)
    return output


def _resize_map(values: np.ndarray, image_size: int) -> np.ndarray:
    if image_size < 16:
        raise ValueError("image_size deve ser pelo menos 16.")
    image = Image.fromarray(np.asarray(values, dtype=np.float32))
    resized = image.resize((image_size, image_size), resample=Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32)


def stft_image(
    signal: np.ndarray,
    *,
    image_size: int = 96,
    nperseg: int = 64,
    noverlap: int = 48,
) -> np.ndarray:
    """Return a resized log-magnitude STFT image (frequency x time)."""
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 1 or signal.size < nperseg:
        raise ValueError("signal deve ser 1-D e ter pelo menos nperseg amostras.")
    _, _, coefficients = stft(
        signal, nperseg=nperseg, noverlap=noverlap, boundary=None, padded=False
    )
    magnitude = np.log1p(np.abs(coefficients)).astype(np.float32)
    return _resize_map(_normalize_map(magnitude), image_size)


def wavelet_image(
    signal: np.ndarray,
    *,
    image_size: int = 96,
    scales: int = 48,
    wavelet: str = "morl",
) -> np.ndarray:
    """Return a resized log-magnitude continuous-wavelet scalogram."""
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 1 or signal.size < 8:
        raise ValueError("signal deve ser 1-D e conter ao menos 8 amostras.")
    if scales < 2:
        raise ValueError("scales deve ser pelo menos 2.")
    coefficients, _ = pywt.cwt(signal, np.arange(1, scales + 1), wavelet)
    scalogram = np.log1p(np.abs(coefficients)).astype(np.float32)
    return _resize_map(_normalize_map(scalogram), image_size)


def ctm_image(
    signal: np.ndarray,
    *,
    image_size: int = 96,
    span: int = 48,
    measure: str = "amplitude",
) -> np.ndarray:
    """Return a square CTM image, with the triangular support centered.

    Valid cells are normalized independently of the zero-valued triangular padding.
    The selected measure and the current CTM integral definition must be reported.
    """
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 1 or signal.size < 2:
        raise ValueError("signal deve ser 1-D e conter ao menos 2 amostras.")
    if measure not in {"amplitude", "energy"}:
        raise ValueError("measure deve ser 'amplitude' ou 'energy'.")
    if span < 2:
        raise ValueError("span deve ser pelo menos 2.")

    values = signal if measure == "amplitude" else signal**2
    points = list(zip(np.arange(signal.size, dtype=np.float64), values))
    grid = build_ctm_grid(points, span)
    triangular = np.zeros((span, span), dtype=np.float32)
    mask = np.zeros((span, span), dtype=bool)
    for row, values_row in enumerate(grid):
        for col, value in enumerate(values_row):
            if value is not None:
                triangular[row, col] = value
                mask[row, col] = True
    normalized = _normalize_map(triangular, valid_mask=mask)
    # Center the lower-triangular CTM support in a square canvas for ResNet.
    canvas = np.zeros_like(normalized)
    for row in range(span):
        columns = np.flatnonzero(mask[row])
        if columns.size:
            target_start = (span - columns.size) // 2
            canvas[row, target_start : target_start + columns.size] = normalized[row, columns]
    return _resize_map(canvas, image_size)


def encode_image(
    signal: np.ndarray,
    representation: str,
    *,
    image_size: int = 96,
    ctm_span: int = 48,
    ctm_measure: str = "amplitude",
) -> np.ndarray:
    """Dispatch one signal window to a 2-D float32 image in [0, 1]."""
    if representation == "stft":
        image = stft_image(signal, image_size=image_size)
    elif representation == "wavelet":
        image = wavelet_image(signal, image_size=image_size)
    elif representation == "ctm":
        image = ctm_image(
            signal, image_size=image_size, span=ctm_span, measure=ctm_measure
        )
    else:
        raise ValueError(f"Representação desconhecida: {representation}")
    return np.ascontiguousarray(image, dtype=np.float32)

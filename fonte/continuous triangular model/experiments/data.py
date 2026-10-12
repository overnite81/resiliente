"""MIT-BIH loading and patient-grouped ECG window extraction."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import wfdb


NORMAL_SYMBOLS = frozenset({"N"})
BEAT_SYMBOLS = frozenset(
    {
        "N",
        "L",
        "R",
        "B",
        "A",
        "a",
        "J",
        "S",
        "V",
        "F",
        "e",
        "j",
        "E",
        "/",
        "f",
        "Q",
        "?",
    }
)


@dataclass(frozen=True)
class ECGWindowDataset:
    """Fixed-length ECG windows and metadata required for grouped evaluation."""

    signals: np.ndarray
    labels: np.ndarray
    groups: np.ndarray
    record_ids: np.ndarray
    annotation_samples: np.ndarray

    def __post_init__(self):
        size = len(self.signals)
        fields = (self.labels, self.groups, self.record_ids, self.annotation_samples)
        if any(len(field) != size for field in fields):
            raise ValueError("Todos os campos do dataset devem ter o mesmo tamanho.")
        if self.signals.ndim != 2:
            raise ValueError("signals deve ter formato (n_janelas, n_amostras).")



def _normalize_window(window: np.ndarray) -> np.ndarray:
    window = np.asarray(window, dtype=np.float64)
    centered = window - np.mean(window)
    scale = np.std(centered)
    if scale == 0:
        return centered
    return centered / scale



def extract_windows_from_signal(
    signal: Sequence[float],
    annotation_samples: Sequence[int],
    annotation_symbols: Sequence[str],
    *,
    record_id: str,
    window_samples: int = 360,
    normal_symbols: Iterable[str] = NORMAL_SYMBOLS,
    beat_symbols: Iterable[str] = BEAT_SYMBOLS,
) -> ECGWindowDataset:
    """Extract normalized beat-centered windows from one ECG signal.

    The interval is centered on each beat annotation. Windows touching the signal
    boundary are discarded instead of padded, keeping every representation based on
    the same observed samples.
    """
    if window_samples <= 0 or window_samples % 2:
        raise ValueError("window_samples deve ser positivo e par.")

    signal_array = np.asarray(signal, dtype=np.float64)
    if signal_array.ndim != 1:
        raise ValueError("signal deve ser unidimensional.")
    if len(annotation_samples) != len(annotation_symbols):
        raise ValueError("Amostras e símbolos das anotações devem ter o mesmo tamanho.")

    normal_symbols = set(normal_symbols)
    beat_symbols = set(beat_symbols)
    half_window = window_samples // 2
    signals = []
    labels = []
    accepted_samples = []

    for sample, symbol in zip(annotation_samples, annotation_symbols):
        if symbol not in beat_symbols:
            continue
        start = int(sample) - half_window
        end = start + window_samples
        if start < 0 or end > len(signal_array):
            continue
        signals.append(_normalize_window(signal_array[start:end]))
        labels.append(0 if symbol in normal_symbols else 1)
        accepted_samples.append(int(sample))

    if not signals:
        raise ValueError(f"Nenhuma janela válida foi encontrada para o registro {record_id}.")

    count = len(signals)
    return ECGWindowDataset(
        signals=np.asarray(signals, dtype=np.float64),
        labels=np.asarray(labels, dtype=np.int8),
        groups=np.full(count, record_id, dtype=object),
        record_ids=np.full(count, record_id, dtype=object),
        annotation_samples=np.asarray(accepted_samples, dtype=np.int64),
    )



def load_mitdb_record(
    record_id: str,
    *,
    window_samples: int = 360,
    channel: int = 0,
    data_dir: str | Path | None = None,
    pn_dir: str = "mitdb",
) -> ECGWindowDataset:
    """Load one MIT-BIH record and extract annotated beat windows.

    If ``data_dir`` is omitted, WFDB fetches the record from PhysioNet. Otherwise,
    ``data_dir / record_id`` is used as the local WFDB record path.
    """
    if data_dir is None:
        record = wfdb.rdrecord(record_id, pn_dir=pn_dir)
        annotation = wfdb.rdann(record_id, "atr", pn_dir=pn_dir)
    else:
        record_path = str(Path(data_dir) / record_id)
        record = wfdb.rdrecord(record_path)
        annotation = wfdb.rdann(record_path, "atr")

    if record.p_signal is None:
        raise ValueError(f"O registro {record_id} não possui sinal físico disponível.")
    if channel < 0 or channel >= record.p_signal.shape[1]:
        raise ValueError(f"Canal inválido: {channel}.")

    return extract_windows_from_signal(
        record.p_signal[:, channel],
        annotation.sample,
        annotation.symbol,
        record_id=record_id,
        window_samples=window_samples,
    )



def combine_datasets(datasets: Sequence[ECGWindowDataset]) -> ECGWindowDataset:
    """Combine records while preserving their grouping metadata."""
    if not datasets:
        raise ValueError("É necessário fornecer pelo menos um dataset.")
    return ECGWindowDataset(
        signals=np.concatenate([dataset.signals for dataset in datasets]),
        labels=np.concatenate([dataset.labels for dataset in datasets]),
        groups=np.concatenate([dataset.groups for dataset in datasets]),
        record_ids=np.concatenate([dataset.record_ids for dataset in datasets]),
        annotation_samples=np.concatenate(
            [dataset.annotation_samples for dataset in datasets]
        ),
    )



def load_mitdb(
    record_ids: Sequence[str],
    *,
    window_samples: int = 360,
    channel: int = 0,
    data_dir: str | Path | None = None,
    pn_dir: str = "mitdb",
) -> ECGWindowDataset:
    """Load and combine a fixed, explicit list of MIT-BIH records."""
    if not record_ids:
        raise ValueError("record_ids não pode ser vazio.")
    records = [
        load_mitdb_record(
            record_id,
            window_samples=window_samples,
            channel=channel,
            data_dir=data_dir,
            pn_dir=pn_dir,
        )
        for record_id in record_ids
    ]
    return combine_datasets(records)

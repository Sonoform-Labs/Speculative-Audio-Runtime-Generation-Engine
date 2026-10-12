from __future__ import annotations

import numpy as np

RMS_BY_INTENSITY = {"soft": 0.06, "normal": 0.10, "hard": 0.16}
PEAK_DBFS_BY_INTENSITY = {"soft": -6.0, "normal": -1.0, "hard": -0.5}
DEFAULT_TARGET_RMS = 0.10
PEAK_CEILING = 0.97


def rms(x) -> float:
    a = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(a * a))) if a.size else 0.0


def target_for(intensity: str) -> float:
    return RMS_BY_INTENSITY.get(intensity, DEFAULT_TARGET_RMS)


def normalize_loudness(x, target_rms: float = DEFAULT_TARGET_RMS,
                       peak_ceiling: float = PEAK_CEILING):
    a = np.asarray(x, dtype=np.float32)
    if a.size == 0:
        return a
    r = rms(a)
    if r > 1e-9:
        a = a * np.float32(target_rms / r)
    peak = float(np.max(np.abs(a)))
    if peak > peak_ceiling:
        a = a * np.float32(peak_ceiling / peak)
    return a.astype(np.float32)


def peak_for(intensity: str) -> float:
    return PEAK_DBFS_BY_INTENSITY.get(intensity, PEAK_DBFS_BY_INTENSITY["normal"])


def normalize_peak(x, peak_dbfs: float = -1.0):
    a = np.asarray(x, dtype=np.float32)
    if a.size == 0:
        return a
    peak = float(np.max(np.abs(a)))
    if peak > 1e-9:
        a = a * np.float32(10.0 ** (peak_dbfs / 20.0) / peak)
    return a.astype(np.float32)

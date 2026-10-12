from __future__ import annotations

import numpy as np

from .amp import family_for

WIDTH = {"blast": 0.45, "impact": 0.35, "step": 0.2, "handling": 0.25}
DEFAULT_WIDTH = 0.35
_KERNELS = {}


def _velvet(sr: int, length_s: float = 0.02, density: float = 1500.0, seed: int = 11) -> np.ndarray:
    k = _KERNELS.get(sr)
    if k is not None:
        return k
    rng = np.random.default_rng(seed)
    n = max(8, int(length_s * sr))
    grid = max(1, int(sr / density))
    h = np.zeros(n)
    for start in range(0, n - grid, grid):
        h[start + int(rng.integers(0, grid))] = rng.choice((-1.0, 1.0)) * np.exp(-3.0 * start / n)
    h /= np.sqrt(np.sum(h ** 2)) or 1.0
    _KERNELS[sr] = h
    return h


def widen(audio, sr: int, event: str = "", width: float = 1.0) -> np.ndarray:
    from scipy.signal import butter, fftconvolve, sosfilt
    a = np.asarray(audio, dtype=np.float64)
    if a.ndim == 1:
        mid, side = a, np.zeros_like(a)
    else:
        mid = a.mean(axis=1)
        side = (a[:, 0] - a[:, 1]) * 0.5
    amount = WIDTH.get(family_for(event), DEFAULT_WIDTH) * float(width)
    if amount > 0 and np.any(mid):
        d = fftconvolve(mid, _velvet(sr), mode="full")[:mid.shape[0]]
        d = sosfilt(butter(2, 250.0, btype="highpass", fs=sr, output="sos"), d)
        side = side + amount * d
    return np.stack([mid + side, mid - side], axis=1).astype(np.float32)

from __future__ import annotations

import numpy as np

FAMILIES = {
    "blast": {"words": ("shoot", "shot", "gunshot", "fire", "explosion", "explode", "blast", "detonat"),
              "punch_db": 6.0, "drive": 2.4},
    "impact": {"words": ("impact", "hit", "break", "smash", "shatter", "slam", "punch", "strike", "land", "crash",
                         "knock", "clang"),
               "punch_db": 4.0, "drive": 1.7},
    "step": {"words": ("footstep", "step", "jump", "walk", "run", "stomp"), "punch_db": 2.5, "drive": 1.25},
    "handling": {"words": ("reload", "equip", "holster", "click", "cock", "rack", "mag"), "punch_db": 3.0, "drive": 1.0},
}
DEFAULT = {"punch_db": 2.0, "drive": 1.0}
INTENSITY_SCALE = {"soft": 0.5, "normal": 1.0, "hard": 1.3}


def family_for(event: str) -> str | None:
    e = (event or "").lower()
    for name, f in FAMILIES.items():
        if any(w in e for w in f["words"]):
            return name
    return None


def _smooth(x: np.ndarray, sr: int, tau_s: float) -> np.ndarray:
    from scipy.signal import lfilter
    a = float(np.exp(-1.0 / max(1.0, tau_s * sr)))
    return lfilter([1.0 - a], [1.0, -a], x)


def amplify(x, sr: int, event: str = "", intensity: str = "normal", punch_scale: float = 1.0, drive: bool = True,
            drive_cap: float | None = None) -> np.ndarray:
    y = np.asarray(x, dtype=np.float64)
    if y.size == 0 or not np.any(y):
        return y.astype(np.float32)
    fam = FAMILIES.get(family_for(event), DEFAULT)
    k = INTENSITY_SCALE.get(intensity, 1.0)
    gmax = 10.0 ** (fam["punch_db"] * k * punch_scale / 20.0)
    a = np.abs(y) if y.ndim == 1 else np.abs(y).mean(axis=1)
    fast = _smooth(a, sr, 0.001)
    slow = _smooth(a, sr, 0.03)
    gain = np.clip(fast / (slow + 1e-9), 1.0, gmax)
    gain = _smooth(gain, sr, 0.002)
    y = y * (gain if y.ndim == 1 else gain[:, None])
    d = 1.0 + (fam["drive"] - 1.0) * k
    if drive_cap is not None:
        d = min(d, float(drive_cap))
    if drive and d > 1.0:
        peak = float(np.max(np.abs(y))) or 1.0
        y = np.tanh(d * y / peak) / np.tanh(d) * peak
    return y.astype(np.float32)

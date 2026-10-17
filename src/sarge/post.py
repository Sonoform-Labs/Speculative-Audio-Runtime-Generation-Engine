from __future__ import annotations

import numpy as np

from .amp import amplify
from .loudness import _sos, dc_highpass, normalize_level, normalize_peak, peak_for

AMP_MODES = ("off", "full", "punch")
LEVELS = ("peak", "loudness")


def envelope(x, sr: int, win_s: float = 0.01) -> np.ndarray:
    m = np.asarray(x, dtype=np.float64)
    m = m if m.ndim == 1 else m.mean(axis=1)
    win = max(1, int(win_s * sr))
    if m.shape[0] <= win:
        return np.zeros(0)
    c = np.empty(m.shape[0] + 1)
    c[0] = 0.0
    np.cumsum(m * m, out=c[1:])
    return np.sqrt(np.maximum(c[win:] - c[:-win], 0.0) / win)


def decay_time(x, sr: int, drop_db: float = 40.0) -> float:
    env = envelope(x, sr)
    if env.size == 0 or not np.any(env):
        return 0.0
    i0 = int(np.argmax(env))
    alive = np.flatnonzero(env >= env[i0] * 10.0 ** (-drop_db / 20.0))
    return (int(alive[-1]) - i0) / sr


def decay_cap(y, sr: int, t40_s: float, hold_s: float = 0.015, slack: float = 1.25, trim_db: float = -50.0,
              fade_s: float = 0.02) -> np.ndarray:
    a = np.asarray(y, dtype=np.float32)
    env = envelope(a, sr)
    if t40_s is None or t40_s <= 0 or env.size == 0 or not np.any(env):
        return a
    i0 = int(np.argmax(env))
    top = float(env[i0])
    db = 20.0 * np.log10(np.maximum(env, top * 1e-6) / top)
    alive = np.flatnonzero(db >= -40.0)
    if (int(alive[-1]) - i0) / sr <= slack * t40_s:
        return a
    s = i0 + int(min(hold_s, 0.5 * t40_s) * sr)
    t = i0 + int(t40_s * sr)
    tail = np.arange(t, env.shape[0])
    slope = float(np.max((db[tail] + 40.0) / ((tail - s) / sr)))
    if slope <= 0.0:
        return a
    g = 10.0 ** (-slope * np.maximum(np.arange(a.shape[0]) - s, 0) / sr / 20.0)
    out = (a * (g if a.ndim == 1 else g[:, None])).astype(np.float32)
    e2 = envelope(out, sr)
    keep = np.flatnonzero(e2 >= top * 10.0 ** (trim_db / 20.0))
    win = max(1, int(0.01 * sr))
    end = min(out.shape[0], int(keep[-1]) + win + int(fade_s * sr)) if keep.size else out.shape[0]
    out = out[:end].copy()
    fo = min(int(fade_s * sr), out.shape[0] // 2)
    if fo > 0:
        out[-fo:] *= np.linspace(1.0, 0.0, fo, dtype=np.float32).reshape((-1,) if out.ndim == 1 else (-1, 1))
    return out


def lowpass(y, sr: int, hz: float) -> np.ndarray:
    from scipy.signal import butter, sosfilt
    hz = min(float(hz), 0.45 * sr)
    sos = _sos("lp%g" % hz, sr, lambda: butter(2, hz, btype="lowpass", fs=sr, output="sos"))
    return sosfilt(sos, np.asarray(y, dtype=np.float64), axis=0).astype(np.float32)


def process(y, sr: int, event: str = "", intensity: str = "normal", *, level: str = "peak", amp_mode: str = "off",
            stereo: bool = False, width: float = 1.0, trim: str = "natural", duration_s: float = 0.6, shape=None,
            stats: dict | None = None) -> np.ndarray:
    from .generators.stable_audio import _extract, _natural
    sh = shape or {}
    a = dc_highpass(np.ascontiguousarray(y, dtype=np.float32), sr).astype(np.float32)
    if trim == "natural":
        a = _natural(a, sr, floor_db=float(sh.get("floor_db", -55.0)))
    elif trim != "none":
        a = _extract(a, sr, duration_s)
    if sh.get("lowpass_hz"):
        a = lowpass(a, sr, float(sh["lowpass_hz"]))
    if amp_mode in ("full", "punch"):
        a = amplify(a, sr, event, "normal" if level == "loudness" else intensity,
                    punch_scale=float(sh.get("punch_scale", 1.0)), drive=amp_mode == "full", drive_cap=sh.get("drive"))
    if stereo:
        from .stereo import widen
        a = widen(a, sr, event, width)
    if level == "loudness":
        a, st = normalize_level(a, sr, intensity)
        if stats is not None:
            stats.update(st)
    elif level == "rms":
        from .loudness import normalize_loudness, target_for
        a = normalize_loudness(a, target_rms=target_for(intensity))
    else:
        a = normalize_peak(a, peak_for(intensity))
    if sh.get("t40_s"):
        a = decay_cap(a, sr, float(sh["t40_s"]))
    return np.asarray(a, dtype=np.float32)

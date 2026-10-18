from __future__ import annotations

import math

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
AMP_MODES = ("full", "punch", "auto")
AUTO = {"punch_db": (2.0, 6.0), "drive": (1.0, 2.4), "width": (0.2, 0.45), "attack_s": (0.005, 0.06),
        "crest_db": (10.0, 18.0, 24.0, 32.0), "flatness": (0.2, 0.7), "t40_s": (0.15, 1.0), "body_db": -20.0,
        "fade_db": 10.0, "decay_slack": 0.02, "decay_floor_s": 0.002}


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


def _mono(x) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    return a if a.ndim == 1 else a.mean(axis=1)


def _ramp(v: float, lo: float, hi: float) -> float:
    return float(min(1.0, max(0.0, (v - lo) / (hi - lo))))


def _rms_env(m: np.ndarray, sr: int, win_s: float) -> np.ndarray:
    from .post import envelope
    return envelope(m, sr, win_s)


def flatness(m: np.ndarray, sr: int, n: int = 2048) -> float:
    s = m if m.shape[0] >= n else np.pad(m, (0, n - m.shape[0]))
    w = np.hanning(n)
    p = np.mean([np.abs(np.fft.rfft(s[i:i + n] * w)) ** 2 for i in range(0, s.shape[0] - n + 1, n // 2)], axis=0)
    f = np.fft.rfftfreq(n, 1.0 / sr)
    flats, energy = [], []
    lo = 125.0
    while lo * 2.0 <= min(16000.0, 0.45 * sr):
        band = p[(f >= lo) & (f < 2.0 * lo)] + 1e-20
        flats.append(float(np.exp(np.mean(np.log(band))) / np.mean(band)))
        energy.append(float(band.sum()))
        lo *= 2.0
    e = np.asarray(energy)
    return float(np.dot(flats, e) / e.sum()) if e.size and e.sum() > 0 else 0.0


def features(x, sr: int) -> dict:
    from .post import decay_time
    m = _mono(x)
    env = _rms_env(m, sr, 0.002)
    if env.size == 0 or not np.any(env):
        return {"attack_s": 0.0, "crest_db": 0.0, "flatness": 0.0, "t40_s": 0.0}
    top = float(env.max())
    on = int(np.flatnonzero(env >= 0.1 * top)[0])
    first = env[on:on + max(1, int(0.1 * sr))]
    attack = int(np.flatnonzero(first >= 0.7 * float(first.max()))[0]) / sr
    end = int(np.flatnonzero(env >= 0.01 * top)[-1]) + max(1, int(0.002 * sr))
    body = m[on:end]
    rms = float(np.sqrt(np.mean(body ** 2))) if body.size else 0.0
    crest = 20.0 * math.log10(float(np.max(np.abs(body))) / rms) if rms > 0 else 0.0
    return {"attack_s": attack, "crest_db": crest, "flatness": flatness(body[:sr], sr), "t40_s": decay_time(m, sr)}


def auto_params(x, sr: int, feats: dict | None = None) -> dict:
    f = feats or features(x, sr)
    a_lo, a_hi = AUTO["attack_s"]
    fast = 1.0 - _ramp(math.log(max(f["attack_s"], 1e-4)), math.log(a_lo), math.log(a_hi))
    c0, c1, c2, c3 = AUTO["crest_db"]
    peaky = _ramp(f["crest_db"], c0, c1) * (1.0 - _ramp(f["crest_db"], c2, c3))
    noisy = _ramp(f["flatness"], *AUTO["flatness"])
    impulsive = fast * peaky
    p_lo, p_hi = AUTO["punch_db"]
    d_lo, d_hi = AUTO["drive"]
    w_lo, w_hi = AUTO["width"]
    return dict(f, impulsive=impulsive, noisy=noisy, punch_db=p_lo + (p_hi - p_lo) * impulsive,
                drive=d_lo + (d_hi - d_lo) * impulsive * noisy,
                width=w_lo + (w_hi - w_lo) * _ramp(f["t40_s"], *AUTO["t40_s"]))


def auto_width(x, sr: int) -> float:
    w_lo, w_hi = AUTO["width"]
    from .post import decay_time
    return w_lo + (w_hi - w_lo) * _ramp(decay_time(_mono(x), sr), *AUTO["t40_s"])


def gated_drive(y: np.ndarray, sr: int, d: float) -> np.ndarray:
    from .post import decay_time
    m = _mono(y)
    win = max(1, int(0.005 * sr))
    env = _rms_env(m, sr, 0.005)
    if env.size == 0 or not np.any(env):
        return y
    env = np.concatenate([np.full(win // 2, env[0]), env, np.full(m.shape[0] - env.shape[0] - win // 2, env[-1])])
    floor = float(env.max()) * 10.0 ** ((AUTO["body_db"] - AUTO["fade_db"]) / 20.0)
    end = min(m.shape[0], int(np.flatnonzero(env > floor)[-1]) + 1 + 8 * win)
    db = 20.0 * np.log10(np.maximum(env[:end], 1e-12) / float(env.max()))
    w = _smooth(np.clip((db - (AUTO["body_db"] - AUTO["fade_db"])) / AUTO["fade_db"], 0.0, 1.0), sr, 0.005)
    peak = float(np.max(np.abs(y))) or 1.0
    head = y[:end]
    sat = np.tanh(d * head / peak) / np.tanh(d) * peak
    out = y.copy()
    out[:end] = head + (sat - head) * (w if y.ndim == 1 else w[:, None])
    m_out = m.copy()
    m_out[:end] = _mono(out[:end])
    if decay_time(m_out, sr) > decay_time(m, sr) * (1.0 + AUTO["decay_slack"]) + AUTO["decay_floor_s"]:
        return y
    return out


def amplify(x, sr: int, event: str = "", intensity: str = "normal", punch_scale: float = 1.0, drive: bool = True,
            drive_cap: float | None = None, auto: bool = False) -> np.ndarray:
    y = np.asarray(x, dtype=np.float64)
    if y.size == 0 or not np.any(y):
        return y.astype(np.float32)
    fam = auto_params(y, sr) if auto else FAMILIES.get(family_for(event), DEFAULT)
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
        if auto:
            y = gated_drive(y, sr, d)
        else:
            peak = float(np.max(np.abs(y))) or 1.0
            y = np.tanh(d * y / peak) / np.tanh(d) * peak
    return y.astype(np.float32)

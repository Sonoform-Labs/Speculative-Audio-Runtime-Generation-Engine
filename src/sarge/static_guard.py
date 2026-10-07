from __future__ import annotations

import hashlib

import numpy as np

try:
    from scipy import fft as _fft
except Exception:
    _fft = None

BROAD_FLAT = 0.12
ONSET_RISE_DB = 9.0
HOLD_DB = -8.0
TAIL_DB = -20.0
BED_FLAT = 0.25
BED_FLOOR_DB = -15.0
AFTER_GAP_S = 0.15
AFTER_MIN_S = 0.15
AFTER_FLAT = 0.15
AFTER_SPREAD_DB = 5.0
AFTER_LEVEL_DB = -32.0
AFTER_DRIFT_DB = 3.0
LOW_HZ = 50.0
FADE_S = 0.05
MIN_S = 0.1
SILENT_RMS = 1e-6
NPERSEG = 2048
NOVERLAP = 1536
_MASK = 0x7FFFFFFF
_RETRY_BIT = 1 << 31
_hann = {}
_QUIET = {"after_flat": 0.0, "after_level_db": -120.0, "after_spread_db": 120.0, "after_drift_db": 0.0}


def _db(a) -> float:
    return float(20.0 * np.log10(max(float(a), 1e-12)))


def _window(n: int) -> np.ndarray:
    w = _hann.get(n)
    if w is None:
        w = (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)).astype(np.float32)
        _hann[n] = w
    return w


def flatness(x, sr: int) -> float:
    x = np.asarray(x, dtype=np.float32)
    n = len(x)
    seg = min(NPERSEG, n)
    step = seg - min(NOVERLAP, n * 3 // 4)
    w = _window(seg)
    frames = np.lib.stride_tricks.sliding_window_view(x, seg)[::step]
    if _fft is not None:
        spec = _fft.rfft(frames * w, axis=1)
    else:
        spec = np.fft.rfft(frames.astype(np.float64) * w, axis=1)
    ri = spec.view(np.float32 if spec.dtype == np.complex64 else np.float64)
    pw = (ri * ri).sum(axis=0)
    p = (pw[0::2] + pw[1::2]).astype(np.float64)
    p *= 1.0 / (len(frames) * sr * float(np.square(w, dtype=np.float64).sum()))
    if seg % 2:
        p[1:] *= 2.0
    else:
        p[1:-1] *= 2.0
    b = p[np.arange(len(p)) * (sr / seg) >= LOW_HZ] + 1e-30
    return float(np.exp(np.mean(np.log(b))) / np.mean(b))


def _env(c: np.ndarray, n: int, win: int, hop: int) -> np.ndarray:
    if n < win:
        return np.sqrt(np.array([c[n] / win]) + 1e-20)
    s = np.arange(0, n - win + 1, hop)
    return np.sqrt(np.maximum(c[s + win] - c[s], 0.0) / win + 1e-20)


def static_metrics(samples, sr: int) -> dict:
    sr = int(sr)
    x = np.asarray(samples, dtype=np.float32)
    n = len(x)
    if n == 0 or n < MIN_S * sr:
        return {"flat": 0.0, "rise_db": 120.0, "post25_db": -120.0, "tail_db": -120.0, "floor10_db": -120.0,
                **_QUIET, "rule": "", "reason": "short"}
    if not np.isfinite(x).all():
        return {"flat": float("inf"), "rise_db": 0.0, "post25_db": 0.0, "tail_db": 0.0, "floor10_db": 0.0,
                **_QUIET, "rule": "nonfinite", "reason": "nonfinite"}
    x64 = x.astype(np.float64)
    c = np.empty(n + 1)
    c[0] = 0.0
    np.cumsum(x64 * x64, out=c[1:])
    hop = max(1, int(0.001 * sr))
    w10 = max(1, int(0.01 * sr))
    e10 = _env(c, n, w10, hop)
    p10 = int(np.argmax(e10))
    peak = float(e10[p10])
    if peak < SILENT_RMS:
        return {"flat": 0.0, "rise_db": 0.0, "post25_db": 0.0, "tail_db": 0.0, "floor10_db": 0.0,
                **_QUIET, "rule": "", "reason": "silent"}
    e5 = _env(c, n, max(1, int(0.005 * sr)), hop)
    p5 = int(np.argmax(e5))
    pk5 = float(e5[p5])
    rise = _db(pk5 / float(e5[max(0, p5 - 25): p5 + 1].min()))
    post = e5[p5 + 1: p5 + 26]
    post25 = _db(float(np.median(post)) / pk5) if len(post) else -120.0
    fo = int(FADE_S * sr)
    fade0 = max(p10 + 1, (n - fo - int(0.01 * sr)) // hop)
    core = e10[5:fade0]
    if len(core) < 3:
        core = e10
    floor10 = _db(float(np.percentile(core, 10)) / peak)
    tail = x64[max(0, n - fo - int(0.1 * sr)): n - fo]
    tail_db = _db(np.sqrt(np.mean(tail * tail)) / peak) if len(tail) else -120.0
    after = dict(_QUIET)
    a = p10 * hop + int(AFTER_GAP_S * sr)
    b = n - fo
    if b - a >= int(AFTER_MIN_S * sr):
        seg = e10[-(-a // hop): (b - w10) // hop + 1]
        if len(seg) >= 3:
            sd = 20.0 * np.log10(seg)
            after["after_spread_db"] = float(np.percentile(sd, 90) - np.percentile(sd, 10))
        mid = a + (b - a) // 2
        after["after_level_db"] = _db(np.sqrt(max(c[b] - c[a], 0.0) / (b - a)) / peak)
        after["after_drift_db"] = (_db(np.sqrt(max(c[b] - c[mid], 0.0) / (b - mid)))
                                   - _db(np.sqrt(max(c[mid] - c[a], 0.0) / (mid - a))))
    flat = flatness(x, sr)
    hiss = flat >= BROAD_FLAT and rise <= ONSET_RISE_DB and (post25 >= HOLD_DB or tail_db >= TAIL_DB)
    bed = flat >= BED_FLAT and floor10 >= BED_FLOOR_DB
    steady = (after["after_spread_db"] <= AFTER_SPREAD_DB and after["after_level_db"] >= AFTER_LEVEL_DB
              and abs(after["after_drift_db"]) <= AFTER_DRIFT_DB)
    if steady and not (hiss or bed):
        after["after_flat"] = flatness(x[a:b], sr)
        steady = after["after_flat"] >= AFTER_FLAT
    else:
        steady = False
    rule = "hiss" if hiss else ("bed" if bed else ("steady" if steady else ""))
    return {"flat": flat, "rise_db": rise, "post25_db": post25, "tail_db": tail_db, "floor10_db": floor10,
            **after, "rule": rule, "reason": ""}


def is_static(samples, sr: int):
    m = static_metrics(samples, sr)
    return bool(m["rule"]), m


def retry_seed(seed: int, attempt: int) -> int:
    h = hashlib.md5(("static_retry|%d|%d" % (int(seed), int(attempt))).encode()).hexdigest()
    return _RETRY_BIT | (int(h[:8], 16) & _MASK)

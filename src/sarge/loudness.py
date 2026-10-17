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


LUFS_BY_INTENSITY = {"soft": -23.0, "normal": -17.0, "hard": -14.0}
CEILING_DBFS = -1.0
MAX_GR_DB = 6.0
SILENCE_LUFS = -70.0
DUAL_MONO_DB = 10.0 * np.log10(2.0)
_SOS = {}


def _shelf_and_highpass(sr: int) -> np.ndarray:
    out = []
    for kind, g, q, fc in (("shelf", 4.0, 1.0 / np.sqrt(2.0), 1500.0), ("hp", 0.0, 0.5, 38.0)):
        a = 10.0 ** (g / 40.0)
        w0 = 2.0 * np.pi * fc / sr
        al = np.sin(w0) / (2.0 * q)
        c = np.cos(w0)
        if kind == "shelf":
            b = [a * ((a + 1) + (a - 1) * c + 2 * np.sqrt(a) * al), -2 * a * ((a - 1) + (a + 1) * c),
                 a * ((a + 1) + (a - 1) * c - 2 * np.sqrt(a) * al)]
            d = [(a + 1) - (a - 1) * c + 2 * np.sqrt(a) * al, 2 * ((a - 1) - (a + 1) * c),
                 (a + 1) - (a - 1) * c - 2 * np.sqrt(a) * al]
        else:
            b = [(1 + c) / 2, -(1 + c), (1 + c) / 2]
            d = [1 + al, -2 * c, 1 - al]
        out.append([b[0] / d[0], b[1] / d[0], b[2] / d[0], 1.0, d[1] / d[0], d[2] / d[0]])
    return np.array(out)


def _sos(name: str, sr: int, make):
    k = (name, int(sr))
    s = _SOS.get(k)
    if s is None:
        s = _SOS[k] = make()
    return s


def kweight(x, sr: int) -> np.ndarray:
    from scipy.signal import sosfilt
    return sosfilt(_sos("k", sr, lambda: _shelf_and_highpass(sr)), np.asarray(x, dtype=np.float64), axis=0)


def _power(k: np.ndarray) -> np.ndarray:
    if k.ndim == 1:
        return k * k
    p = k[:, 0] * k[:, 0]
    for i in range(1, k.shape[1]):
        p += k[:, i] * k[:, i]
    return p


def _peak(a: np.ndarray) -> np.ndarray:
    if a.ndim == 1:
        return np.abs(a)
    p = np.abs(a[:, 0])
    for i in range(1, a.shape[1]):
        np.maximum(p, np.abs(a[:, i]), out=p)
    return p


def momentary_max(x, sr: int, win_s: float = 0.4, hop_s: float = 0.005) -> float:
    k = kweight(x, sr)
    p = _power(k)
    hop = max(1, int(round(hop_s * sr)))
    w = max(1, int(round(win_s * sr / hop)))
    nb = max(w, -(-p.shape[0] // hop))
    if nb * hop > p.shape[0]:
        p = np.concatenate([p, np.zeros(nb * hop - p.shape[0])])
    c = np.concatenate([[0.0], np.cumsum(p.reshape(nb, hop).sum(axis=1))])
    s = (c[w:] - c[:-w]).max() / (w * hop)
    return float(-0.691 + 10.0 * np.log10(max(float(s), 1e-20)))


def dc_highpass(x, sr: int, hz: float = 30.0) -> np.ndarray:
    from scipy.signal import butter, sosfilt
    a = np.asarray(x, dtype=np.float64)
    if a.size == 0:
        return a
    return sosfilt(_sos("hp%g" % hz, sr, lambda: butter(2, hz, btype="highpass", fs=sr, output="sos")), a, axis=0)


def _gain_reduction_db(peak, sr: int, thresh: float, lookahead_s: float, release_s: float):
    from scipy.ndimage import maximum_filter1d, uniform_filter1d
    over = np.flatnonzero(peak > thresh)
    if over.size == 0:
        return None
    n = max(1, int(round(lookahead_s * sr)))
    b = max(1, n // 6)
    nl = -(-n // b)
    rel = max(1.0, release_s * sr)
    s0 = max(0, int(over[0]) - 2 * n - 2 * b)
    s1 = min(peak.shape[0], int(over[-1]) + 2 * n + 2 * b + int(8.0 * rel))
    seg = peak[s0:s1]
    nb = -(-seg.shape[0] // b)
    if nb * b > seg.shape[0]:
        seg = np.concatenate([seg, np.zeros(nb * b - seg.shape[0], dtype=seg.dtype)])
    bp = seg.reshape(nb, b).max(axis=1).astype(np.float64)
    size = nl + 3
    ahead = maximum_filter1d(bp, size=size, origin=1 - size // 2, mode="nearest")
    need = np.log(np.maximum(ahead * (1.0 / thresh), 1.0))
    g = uniform_filter1d(need, size=nl + 1, origin=nl // 2, mode="nearest")
    rate = np.arange(nb, dtype=np.float64) * (b / rel)
    z = np.log(np.maximum(g, 1e-30)) + rate
    np.maximum.accumulate(z, out=z)
    z = np.exp(z - rate)
    centers = np.arange(nb, dtype=np.float64) * b + 0.5 * (b - 1)
    out = np.interp(np.arange(s1 - s0, dtype=np.float64), centers, z)
    return s0, out * (20.0 / np.log(10.0))


def limit(x, sr: int, ceiling_dbfs: float = CEILING_DBFS, lookahead_s: float = 0.002, release_s: float = 0.05,
          stats: dict | None = None, gain_db: float = 0.0, post_db: float = 0.0) -> np.ndarray:
    a = np.asarray(x, dtype=np.float32)
    g = 10.0 ** (gain_db / 20.0)
    o = 10.0 ** (post_db / 20.0)
    c = 10.0 ** (ceiling_dbfs / 20.0)
    if stats is not None:
        stats["gr_db"] = 0.0
    y = a * np.float32(g * o)
    if a.size == 0:
        return y
    r = _gain_reduction_db(_peak(a), sr, c / g, lookahead_s, release_s)
    if r is None:
        return y
    s0, gr = r
    if stats is not None:
        stats["gr_db"] = float(gr.max())
    k = np.exp(gr * (-np.log(10.0) / 20.0)).astype(np.float32)
    seg = y[s0:s0 + k.shape[0]]
    seg *= k if a.ndim == 1 else k[:, None]
    np.clip(seg, -np.float32(c * o), np.float32(c * o), out=seg)
    return y


def normalize_level(x, sr: int, intensity: str = "normal", ceiling_dbfs: float = CEILING_DBFS,
                    max_gr_db: float = MAX_GR_DB):
    a = np.asarray(x, dtype=np.float32)
    st = {"lufs_in": float("-inf"), "gain_db": 0.0, "gr_db": 0.0}
    if a.size == 0 or not np.any(a):
        return a.copy(), st
    m = momentary_max(a, sr) + (DUAL_MONO_DB if a.ndim == 1 else 0.0)
    if not m > SILENCE_LUFS:
        st.update(lufs_in=m)
        return a.copy(), st
    pk = 20.0 * np.log10(float(np.max(np.abs(a))))
    gain = min(LUFS_BY_INTENSITY.get(intensity, LUFS_BY_INTENSITY["normal"]) - m, ceiling_dbfs + max_gr_db - pk)
    y = limit(a, sr, ceiling_dbfs, stats=st, gain_db=gain)
    st.update(lufs_in=m, gain_db=gain)
    return y, st

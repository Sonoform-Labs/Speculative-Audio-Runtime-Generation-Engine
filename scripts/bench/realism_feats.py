import os
import sys

BENCH = os.path.join(os.getcwd(), "scripts", "bench")
if not os.path.isfile(os.path.join(BENCH, "_bootstrap.py")):
    BENCH = r"D:\Python\SAGE\scripts\bench"
sys.path.insert(0, BENCH)
TRAIN = os.path.join(os.path.dirname(BENCH), "train")
if TRAIN not in sys.path:
    sys.path.insert(0, TRAIN)

import _bootstrap

import hashlib
import math

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import butter, hilbert, lfilter, resample_poly, sosfiltfilt, welch

import dec_common as dc

SR = 44100
NAN = float("nan")
FOOT_END_S = 0.42
HOP_ENV = 0.25
BAND_MIN_DB = -45.0
COMB_HZ = SR / 256.0
COMB_MAX_HZ = 15000.0
SPARK_LO, SPARK_HI = 3000.0, 12000.0
SPARK_N, SPARK_HOP = 2048, 512
SPARK_DB, SPARK_ACTIVE_DB = 12.0, 30.0
SPARK_PER_OCT = 48
SPARK_PERSIST, SPARK_PERSIST_MIN = 0.6, 5
J5_BANDS = [(100.0, 200.0), (200.0, 400.0), (400.0, 800.0), (800.0, 1600.0), (1600.0, 3200.0), (3200.0, 6400.0),
            (6400.0, 12800.0)]
J5_GROUPS = {"lo": (0, 1, 2), "mid": (3, 4), "hi": (5, 6)}
ENV_SR = 400
MOD_A, MOD_B = (10.0, 32.0), (32.0, 128.0)
TAIL_S = 0.3

J1 = ["crest_db", "plr10_db", "plr50_db", "e10_100", "rise_ms", "fwhm_ms", "pre_db", "low_share"]
J2 = ["n_clicks", "click2_db", "gap_ms"]
J3 = ["hfk_lo", "hfk_hi", "hfk_frame", "spike_rate", "flux_kurt", "sparkle"]
J4 = ["comb172", "comb_any", "flutter"]
J5 = ["ek_%d" % i for i in range(7)] + ["mf_%s_%s" % (g, m) for g in J5_GROUPS for m in ("a", "b")] + \
     ["ecorr_1", "ecorr_3"]
J9 = ["decay_db_s", "late_early_db", "tail_sparkle", "tail_hfk_hi"]
DESC = ["rolloff99", "hf16_db", "clip_frac", "dc", "zero_frac", "t_loudest", "peak", "comb_f0", "two_click"]
ALL = J1 + J2 + J3 + J4 + J5
DIRS = {"crest_db": "lowcap", "plr10_db": "lowcap", "plr50_db": "lowcap", "e10_100": "low", "rise_ms": "high",
        "fwhm_ms": "high", "pre_db": "high", "low_share": "low", "comb172": "high", "comb_any": "high",
        "flutter": "high"}
_SOS = {}


def feats_md5():
    with open(os.path.abspath(__file__), "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


def direction(f):
    return DIRS.get(f, "two")


def sos(kind, fc, order):
    k = (kind, fc if np.isscalar(fc) else tuple(fc), order)
    if k not in _SOS:
        _SOS[k] = butter(order, fc, kind, fs=SR, output="sos")
    return _SOS[k]


def band(x, lo, hi, order=4):
    if hi is None:
        return sosfiltfilt(sos("highpass", lo, order), x)
    if lo is None:
        return sosfiltfilt(sos("lowpass", hi, order), x)
    return sosfiltfilt(sos("bandpass", [lo, hi], order), x)


def energy(x):
    return float(np.mean(np.asarray(x, np.float64) ** 2)) if len(x) else 0.0


def db(r, floor=-120.0):
    return float(10.0 * np.log10(r)) if r > 10 ** (floor / 10.0) else floor


def present(h, tot):
    return tot > 0 and energy(h) > tot * 10 ** (BAND_MIN_DB / 10.0)


def logk(h):
    k = dc.kurt(np.asarray(h, np.float64))
    return math.log(k) if k == k and k > 0 else NAN


def kweight_coefs():
    f0, G, Q = 1681.974450955533, 3.999843853973347, 0.7071752369554196
    K = math.tan(math.pi * f0 / SR)
    Vh = 10 ** (G / 20.0)
    Vb = Vh ** 0.4996667741545416
    a0 = 1.0 + K / Q + K * K
    b1 = [(Vh + Vb * K / Q + K * K) / a0, 2.0 * (K * K - Vh) / a0, (Vh - Vb * K / Q + K * K) / a0]
    a1 = [1.0, 2.0 * (K * K - 1.0) / a0, (1.0 - K / Q + K * K) / a0]
    f0, Q = 38.13547087602444, 0.5003270373238773
    K = math.tan(math.pi * f0 / SR)
    d = 1.0 + K / Q + K * K
    b2 = [1.0, -2.0, 1.0]
    a2 = [1.0, 2.0 * (K * K - 1.0) / d, (1.0 - K / Q + K * K) / d]
    return b1, a1, b2, a2


KW = kweight_coefs()


def kweight(x):
    b1, a1, b2, a2 = KW
    return lfilter(b2, a2, lfilter(b1, a1, np.asarray(x, np.float64)))


def integrated_lufs(x):
    y = kweight(x)
    n = int(0.4 * SR)
    hop = int(0.1 * SR)
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    c = np.concatenate([[0.0], np.cumsum(y * y)])
    st = np.arange(0, len(y) - n + 1, hop)
    ms = (c[st + n] - c[st]) / n
    lk = -0.691 + 10 * np.log10(ms + 1e-30)
    g = ms[lk > -70.0]
    if not len(g):
        return -70.0
    rel = -0.691 + 10 * np.log10(np.mean(g)) - 10.0
    g2 = ms[(lk > -70.0) & (lk > rel)]
    return float(-0.691 + 10 * np.log10(np.mean(g2 if len(g2) else g)))


def sliding_ms(x, n):
    c = np.concatenate([[0.0], np.cumsum(np.asarray(x, np.float64) ** 2)])
    if len(x) < n:
        return np.array([c[-1] / max(1, len(x))])
    return (c[n:] - c[:-n]) / n


def hop_samples():
    return max(1, int(round(HOP_ENV * SR / 1000.0)))


def onset(a):
    h = band(a, 2000.0, None, 4)
    env = dc.env_ms(h, 1.0, HOP_ENV)
    if not len(env) or env.max() <= 1e-9:
        return None, env
    pk = int(np.argmax(env))
    top = env[pk]
    j = pk
    while j > 0 and env[j - 1] >= 0.1 * top:
        j -= 1
    t90 = j + int(np.argmax(env[j:pk + 1] >= 0.9 * top))
    return {"hop": j, "t0": j * hop_samples(), "pk": pk, "rise_ms": (t90 - j) * HOP_ENV}, env


def flux_series(x, n, hop, lo=None, hi=None):
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    M = np.sqrt(dc.np_stft_pow(x, n, hop))
    a, b = dc._bins(n, lo or 0.0, hi)
    return np.maximum(np.diff(M[a:b], axis=1), 0.0).sum(0)


def fwhm_ms(a):
    fl = flux_series(a, 256, 32)
    if not fl.size or fl.max() <= 0:
        return NAN
    p = int(np.argmax(fl))
    half = 0.5 * fl[p]
    lo = p
    while lo > 0 and fl[lo - 1] >= half:
        lo -= 1
    hi = p
    while hi < len(fl) - 1 and fl[hi + 1] >= half:
        hi += 1
    w = (hi - lo + 1) * 32.0 / SR * 1000.0
    return math.log(max(w, 32.0 / SR * 1000.0))


def j1(a, fam):
    out = {k: NAN for k in J1}
    if energy(a) <= 1e-14:
        return out, None
    on, env = onset(a)
    if on is None:
        return out, None
    t0 = on["t0"]
    r = math.sqrt(energy(a))
    out["crest_db"] = 20 * math.log10(float(np.max(np.abs(a))) / r)
    k = kweight(a)
    km = energy(k)
    if km > 0:
        out["plr10_db"] = db(float(sliding_ms(k, int(0.01 * SR)).max()) / km)
        out["plr50_db"] = db(float(sliding_ms(k, int(0.05 * SR)).max()) / km)
    e10 = float(np.sum(a[t0:t0 + int(0.01 * SR)] ** 2))
    e100 = float(np.sum(a[t0:t0 + int(0.1 * SR)] ** 2))
    if e100 > 0:
        out["e10_100"] = db(e10 / e100)
    out["rise_ms"] = math.log(min(max(on["rise_ms"], HOP_ENV), 30.0))
    out["fwhm_ms"] = fwhm_ms(a)
    p0, p1 = max(0, t0 - int(0.02 * SR)), max(0, t0 - int(0.005 * SR))
    if p1 - p0 >= 32 and e10 > 0:
        out["pre_db"] = max(-60.0, db(float(np.sum(a[p0:p1] ** 2)) / e10))
    else:
        out["pre_db"] = -60.0
    if fam in ("gun", "other"):
        seg = a[t0:t0 + int(0.1 * SR)]
        tot = energy(seg)
        if tot > 0:
            out["low_share"] = max(-60.0, db(energy(band(a, None, 250.0, 4)[t0:t0 + int(0.1 * SR)]) / tot))
    return out, on


def j2(a, on):
    out = {"n_clicks": NAN, "click2_db": NAN, "gap_ms": NAN, "two_click": NAN}
    if on is None:
        return out
    env = dc.env_ms(band(a, 2000.0, 16000.0, 4), 1.0, HOP_ENV)
    s = on["hop"]
    e = min(len(env) - 1, s + int(60.0 / HOP_ENV))
    if e - s < 3:
        return out
    seg = env[s:e + 1]
    top = float(seg.max())
    if top <= 0:
        return out
    cand = []
    for i in range(max(1, s), e):
        v = env[i]
        if v < env[i - 1] or v <= env[i + 1] or v < top * 10 ** (-18.0 / 20.0):
            continue
        lo = max(0, i - int(2.0 / HOP_ENV))
        if i > lo and env[lo:i].min() <= 0.3 * v:
            cand.append((v, i))
    cand.sort(reverse=True)
    picked = []
    gap = int(4.0 / HOP_ENV)
    for v, i in cand:
        if all(abs(i - j) >= gap for _v, j in picked):
            picked.append((v, i))
    n = len(picked)
    out["n_clicks"] = float(min(n, 4))
    out["two_click"] = float(n >= 2)
    if n >= 2:
        out["click2_db"] = 20 * math.log10(picked[1][0] / picked[0][0])
        out["gap_ms"] = math.log(max(abs(picked[1][1] - picked[0][1]) * HOP_ENV, HOP_ENV))
    else:
        out["click2_db"] = -30.0
        out["gap_ms"] = math.log(60.0)
    return out


def j3(a):
    out = {k: NAN for k in J3 if k != "sparkle"}
    tot = energy(a)
    if tot <= 1e-14:
        return out
    lo = band(a, 2000.0, 8000.0, 4)
    hi = band(a, 4000.0, 16000.0, 4)
    if present(lo, tot):
        out["hfk_lo"] = logk(lo)
    if present(hi, tot):
        out["hfk_hi"] = logk(hi)
        idx = dc.active_frames(hi)
        hf = dc.hfk_frame(hi, idx)
        out["hfk_frame"] = math.log(hf) if hf == hf and hf > 0 else NAN
        sr_ = dc.spike_rate(hi, idx)
        out["spike_rate"] = math.log1p(sr_) if sr_ == sr_ else NAN
        fl = flux_series(a, 512, 128, 4000.0, 16000.0)
        k = dc.kurt(fl) if fl.size else NAN
        out["flux_kurt"] = math.log(k) if k == k and k > 0 else NAN
    return out


def comb172(a):
    if len(a) < 64:
        return NAN
    n = 8192
    f, p = dc.welch_pow(a, n)
    nn = min(n, len(a))
    step = COMB_HZ * nn / SR
    best = -np.inf
    k = 1
    while True:
        c = int(round(k * step))
        if c + 14 >= len(p) or k * COMB_HZ > COMB_MAX_HZ:
            break
        if c - 12 >= 0:
            nb = np.concatenate([p[c - 12:c - 2], p[c + 3:c + 13]])
            best = max(best, 10 * np.log10((p[c] + 1e-30) / (np.median(nb) + 1e-30)))
        k += 1
    return float(best) if np.isfinite(best) else NAN


F0_GRID = np.arange(60.0, 400.01, 0.5)
_HARM = {}


def harmonics(lo, hi):
    k = (round(lo, 6), round(hi, 6))
    if k not in _HARM:
        owner, pos = [], []
        for i, f0 in enumerate(F0_GRID):
            ks = np.arange(int(math.ceil(lo / f0)), int(math.floor(hi / f0 - 0.5)) + 1)
            owner.append(np.full(len(ks), i))
            pos.append(ks * f0)
        owner = np.concatenate(owner)
        pos = np.concatenate(pos)
        _HARM[k] = (owner, pos, np.bincount(owner, minlength=len(F0_GRID)).astype(np.float64))
    return _HARM[k]


def comb_any(a):
    if len(a) < 2048 or energy(a) <= 1e-14:
        return NAN, NAN
    f, p = welch(np.asarray(a, np.float64), fs=SR, nperseg=min(8192, len(a)), noverlap=min(8192, len(a)) // 2)
    df = f[1] - f[0]
    sel = (f >= 2000.0) & (f <= COMB_MAX_HZ)
    L = 10 * np.log10(p[sel] + 1e-30)
    w = max(3, int(round(400.0 / df)) | 1)
    r = L - median_filter(L, size=w, mode="nearest")
    fs_ = f[sel]
    owner, pos, cnt = harmonics(fs_[0], fs_[-1])
    on = np.bincount(owner, weights=np.interp(pos, fs_, r), minlength=len(F0_GRID))
    off = np.bincount(owner, weights=np.interp(pos + F0_GRID[owner] * 0.5, fs_, r), minlength=len(F0_GRID))
    with np.errstate(invalid="ignore", divide="ignore"):
        sc = np.where(cnt > 0, (on - off) / np.maximum(cnt, 1), np.nan)
    if not np.isfinite(sc).any():
        return NAN, NAN
    j = int(np.nanargmax(sc))
    return float(sc[j] - np.nanmedian(sc)), float(F0_GRID[j])


def j4(a):
    prom, f0 = comb_any(a)
    return {"comb172": comb172(a), "comb_any": prom, "flutter": dc.flutter_538(a) if energy(a) > 1e-14 else NAN,
            "comb_f0": f0}


_GRID = {}


def spark_grid():
    if "g" not in _GRID:
        lo = SPARK_LO * 2 ** (-1.0 / 6.0)
        hi = SPARK_HI * 2 ** (1.0 / 6.0)
        n = int(math.ceil(math.log2(hi / lo) * SPARK_PER_OCT)) + 1
        _GRID["g"] = lo * 2 ** (np.arange(n) / float(SPARK_PER_OCT))
    return _GRID["g"]


def sparkle(a, f0=None):
    if len(a) < SPARK_N or energy(a) <= 1e-14:
        return NAN
    P = dc.np_stft_pow(a, SPARK_N, SPARK_HOP)
    fe = P.sum(0)
    if fe.max() <= 0:
        return NAN
    act = np.flatnonzero(fe >= fe.max() * 10 ** (-SPARK_ACTIVE_DB / 10.0))
    if not len(act):
        return NAN
    df = SR / float(SPARK_N)
    freqs = np.arange(P.shape[0]) * df
    b0, b1 = int(math.ceil(SPARK_LO / df)), int(math.floor(SPARK_HI / df))
    tot = energy(a)
    if not present(band(a, SPARK_LO, SPARK_HI, 4), tot):
        return NAN
    g = spark_grid()
    lg = np.log(g)
    lf = np.log(np.maximum(freqs[b0:b1 + 1], 1.0))
    D = 10 * np.log10(P[:, act] + 1e-30)
    Gd = np.stack([np.interp(g, freqs, D[:, i]) for i in range(D.shape[1])], axis=1)
    bg = median_filter(Gd, size=(2 * (SPARK_PER_OCT // 6) + 1, 1), mode="nearest")
    B = np.stack([np.interp(lf, lg, bg[:, i]) for i in range(bg.shape[1])], axis=1)
    S = D[b0:b1 + 1]
    Sm = D[b0 - 1:b1]
    Sp = D[b0 + 1:b1 + 2]
    pk = (S > Sm) & (S >= Sp) & (S >= B + SPARK_DB)
    bins = np.arange(b0, b1 + 1)
    keep = ~np.isin(bins % 8, (7, 0, 1))
    if f0 is not None and f0 == f0:
        hb = np.round(np.arange(1, int(SPARK_HI / f0) + 2) * f0 / df).astype(int)
        near = np.zeros(len(bins), bool)
        for h in hb:
            near |= np.abs(bins - h) <= 1
        keep &= ~near
    pk &= keep[:, None]
    if pk.shape[1] >= SPARK_PERSIST_MIN:
        pk &= (pk.mean(1) < SPARK_PERSIST)[:, None]
    return float(pk.sum(0).mean())


def env_bands(a):
    tot = energy(a)
    envs, ok = [], []
    for lo, hi in J5_BANDS:
        b = band(a, lo, hi, 4)
        ok.append(present(b, tot))
        e = np.abs(hilbert(b))
        e = band(e, None, 150.0, 4)
        e = resample_poly(e, 4, 441)
        envs.append(np.maximum(e, 0.0) ** 0.3)
    return envs, ok


def mod_frac(e):
    v = e - e.mean()
    if len(v) < 16 or np.allclose(v, 0):
        return NAN, NAN
    S = np.abs(np.fft.rfft(v * np.hanning(len(v)))) ** 2
    fr = np.fft.rfftfreq(len(v), 1.0 / ENV_SR)
    tot = S[fr > 0].sum()
    if tot <= 0:
        return NAN, NAN
    fa = S[(fr >= MOD_A[0]) & (fr < MOD_A[1])].sum() / tot
    fb = S[(fr >= MOD_B[0]) & (fr < MOD_B[1])].sum() / tot
    return math.log(max(fa, 1e-6)), math.log(max(fb, 1e-6))


def corr(a, b):
    if a.std() < 1e-12 or b.std() < 1e-12:
        return NAN
    return float(np.corrcoef(a, b)[0, 1])


def j5(a, foot):
    out = {k: NAN for k in J5}
    if energy(a) <= 1e-14:
        return out
    envs, ok = env_bands(a)
    if foot:
        m = int(FOOT_END_S * ENV_SR)
        envs = [e[:m] for e in envs]
    for i, (e, o) in enumerate(zip(envs, ok)):
        if o and e.std() > 1e-12:
            v = e - e.mean()
            out["ek_%d" % i] = math.log(float(np.mean(v ** 4) / np.mean(v ** 2) ** 2))
    for gname, ids in J5_GROUPS.items():
        fa, fb = [], []
        for i in ids:
            if ok[i]:
                x, y = mod_frac(envs[i])
                if x == x:
                    fa.append(x)
                    fb.append(y)
        if fa:
            out["mf_%s_a" % gname] = float(np.mean(fa))
            out["mf_%s_b" % gname] = float(np.mean(fb))
    for d, name in ((1, "ecorr_1"), (3, "ecorr_3")):
        cs = [corr(envs[i], envs[i + d]) for i in range(7 - d) if ok[i] and ok[i + d]]
        cs = [c for c in cs if c == c]
        if cs:
            out[name] = float(np.mean(cs))
    return out


def j9(a1, wet):
    out = {k: NAN for k in J9}
    tot = energy(a1)
    if tot <= 1e-14:
        return out
    e = a1 * a1
    edc = np.cumsum(e[::-1])[::-1]
    edb = 10 * np.log10(edc / edc[0] + 1e-30)
    i5 = np.flatnonzero(edb <= -5.0)
    i25 = np.flatnonzero(edb <= -25.0)
    if len(i5) and len(i25) and i25[0] > i5[0]:
        out["decay_db_s"] = math.log(20.0 / ((i25[0] - i5[0]) / SR))
    n3 = int(TAIL_S * SR)
    early, late = energy(a1[:n3]) * n3, energy(a1[n3:]) * max(1, len(a1) - n3)
    if early > 0:
        out["late_early_db"] = max(-80.0, db(late / early))
    if wet:
        t = a1[n3:]
        tt = energy(t)
        if tt > 1e-14:
            h = band(t, 4000.0, 16000.0, 4)
            if present(h, tt):
                out["tail_hfk_hi"] = logk(h)
    return out


def descriptive(x5, raw_dc):
    x5 = np.asarray(x5, np.float64)
    out = {"dc": float(raw_dc), "zero_frac": float(np.mean(x5 == 0.0)), "peak": float(np.max(np.abs(x5))) if len(x5)
           else 0.0, "t_loudest": dc.loudest(x5, max(1, int(0.005 * SR))) / SR if len(x5) else NAN}
    P = np.abs(np.fft.rfft(x5)) ** 2
    f = np.fft.rfftfreq(len(x5), 1.0 / SR)
    tot = P.sum()
    if tot > 0:
        c = np.cumsum(P) / tot
        out["rolloff99"] = float(f[int(np.searchsorted(c, 0.99))])
        out["hf16_db"] = db(P[f >= 16000.0].sum() / tot)
    else:
        out["rolloff99"], out["hf16_db"] = NAN, NAN
    pk = out["peak"]
    out["clip_frac"] = float(np.mean(np.abs(x5) >= 0.999 * pk)) if pk > 0 else NAN
    return out


def clip_features(a, x5, raw_dc, fam, foot, wet, win):
    if win == "1.0":
        out = j9(a, wet)
        out.update(j4(a))
        out.update(descriptive(x5, raw_dc))
        return out
    r = a[:int(FOOT_END_S * SR)] if foot else a
    out, on = j1(r, fam)
    out.update(j2(r, on))
    out.update(j3(r))
    out.update(j4(r))
    out.update(j5(a, foot))
    out.update(descriptive(x5, raw_dc))
    return out


def spark_input(a, foot, win):
    if win == "1.0":
        return a[int(TAIL_S * SR):]
    return a[:int(FOOT_END_S * SR)] if foot else a


def audio_md5(a):
    return hashlib.md5(np.ascontiguousarray(a, np.float64).tobytes()).hexdigest()


class Cache:
    def __init__(self, path, tag):
        self.path = path
        self.tag = tag
        self.data = {}
        self.dirty = False
        if os.path.isfile(path):
            try:
                d = np.load(path, allow_pickle=True).item()
                if d.get("_tag") == tag:
                    self.data = d
            except Exception:
                self.data = {}
        self.data["_tag"] = tag

    def get(self, k):
        return self.data.get(k)

    def put(self, k, v):
        self.data[k] = v
        self.dirty = True

    def save(self):
        if self.dirty:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            tmp = self.path + ".tmp.npy"
            np.save(tmp, self.data, allow_pickle=True)
            os.replace(tmp, self.path)
            self.dirty = False


class Panns:
    def __init__(self, threads=4):
        import torch
        torch.set_num_threads(threads)
        from panns_inference import AudioTagging
        self.torch = torch
        self.at = AudioTagging(checkpoint_path=None, device="cpu")

    def embed(self, xs, bs=32):
        out = []
        for i in range(0, len(xs), bs):
            chunk = [resample_poly(np.asarray(x, np.float64), 320, 441).astype(np.float32) for x in xs[i:i + bs]]
            n = max(max(len(c) for c in chunk), 16000)
            batch = np.stack([np.pad(c, (0, n - len(c))) for c in chunk])
            with self.torch.no_grad():
                _cw, emb = self.at.inference(batch)
            e = np.asarray(emb, np.float64)
            e /= np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)
            out.append(e)
        return np.concatenate(out) if out else np.zeros((0, 2048))

import os
import sys

BENCH = os.path.join(os.getcwd(), "scripts", "bench")
if not os.path.isfile(os.path.join(BENCH, "_bootstrap.py")):
    BENCH = r"D:\Python\SAGE\scripts\bench"
sys.path.insert(0, BENCH)

import _bootstrap

import argparse
import csv
import glob
import hashlib
import json
import math
import re
from collections import defaultdict

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter, resample_poly, sosfiltfilt

import vocab_eval as ve
from sarge.generators.stable_audio import _extract
from sarge.loudness import normalize_loudness, target_for
from water_judge import HISS_PER, HISS_SPECTRA, hiss
from vocab_refs import C3GD, FOLEY

ROOT = ve.ROOT
SR = 44100
CHAIN_VERSION = "P1.1"
HELDOUT = os.path.join("data", "refs_heldout", "manifest.json")
REFS_ALL = os.path.join("data", "refs", "manifest.json")
CACHE_V1 = os.path.join(ROOT, "data", "decoder_ft", "cache_v1")
ANCHOR = os.path.join(ROOT, "data", "_gate", "anchor", "raw")
ROUNDTRIP = os.path.join(ROOT, "data", "_gate", "ae_roundtrip")
EXTRA_TYPES = [("bullet_impact", "glass", "glass_break"), ("footstep", "water", "splash"), ("splash", "water", "splash")]
PISTOL = ["hk_P30L", "ruger_lcp_380ACP", "sig_p365", "springfieldd_xd", "steyr_C9A2"]
RIFLE = ["7mm_Mauser", "marlin_1895", "norinco_sks", "savage_223", "savage_mh110", "sw_mp15"]
KEYS = {
    "shoot|pistol": {"family": "gun", "pool": "pistol", "status": "real"},
    "shoot|rifle": {"family": "gun", "pool": "rifle", "status": "real"},
    "break|window": {"family": "glass", "pool": "glass", "status": "real"},
    "bullet_impact|glass": {"family": "glass", "pool": "glass", "status": "proxy"},
    "footstep|gravel": {"family": "footstep", "pool": "gravel", "status": "real"},
    "footstep|stone": {"family": "footstep", "pool": "stone", "status": "real"},
    "footstep|wood": {"family": "footstep", "pool": "wood", "status": "real"},
    "footstep|metal": {"family": "footstep", "pool": None, "status": "pool-free"},
    "splash|water": {"family": "water", "pool": "splash", "status": "real"},
    "footstep|water": {"family": "water", "pool": None, "status": "pool-free"},
    "punch|body": {"family": "other", "pool": "punch", "status": "exploratory"},
    "reload|rifle": {"family": "other", "pool": "reload", "status": "exploratory"},
}
POOLS = {
    "pistol": {"family": "gun", "src": "heldout", "sets": ["c3gd_pistol"], "platforms": PISTOL, "filter": False},
    "rifle": {"family": "gun", "src": "heldout", "sets": ["c3gd_rifle"], "platforms": RIFLE, "filter": False},
    "glass": {"family": "glass", "src": "heldout", "sets": ["fs_glass", "esc_glass_breaking"], "filter": True},
    "gravel": {"family": "footstep", "src": "cache_v1", "bucket": "step_gravel", "filter": True},
    "stone": {"family": "footstep", "src": "cache_v1", "bucket": "step_stone", "filter": True},
    "wood": {"family": "footstep", "src": "cache_v1", "bucket": "step_wood", "filter": True},
    "splash": {"family": "water", "src": "heldout", "sets": ["fs_splash"], "filter": True},
    "punch": {"family": "other", "src": "heldout", "sets": ["fs_punch"], "filter": True},
    "reload": {"family": "other", "src": "refs", "sets": ["fs_gunmech"], "filter": True},
}
FAMILIES = ["gun", "glass", "footstep", "water", "other"]
CORE_FAMILIES = ["gun", "glass", "footstep", "water"]
SURFACES = ["gravel", "stone", "wood"]
MIN_CLIPS, MIN_SOURCES = 20, 4
Q_CLIP, Q_ROLLOFF = 1e-3, 11000.0
J9_AVAIL = 0.8
J9_MIN_AFTER_S = 1.0
CROP_PREV_S, CROP_NEXT_S, CROP_PRE_S, CROP_CUT_S, CROP_MAX = 0.15, 0.5, 0.1, 0.02, 6
GATE_END_S, GATE_FADE_S = 0.45, 0.03
DAMAGE_N = 20
FLAW_SEEDS = 4
D1_GAINS = (0.3, 0.5, 0.7)
D2_TAUS = (1.0, 3.0, 6.0)
D5_SNRS = (30.0, 20.0, 10.0)
NUISANCE = ("rt48", "pre25", "pre35", "polarity", "gain05")
FLAWS = ("clip02", "clip1", "pink40", "lp11")
_SOS = {}


def key_of(et, mat):
    return "%s|%s" % (et.replace(" ", "_"), mat)


def fname(key):
    et, mat = key.split("|")
    return "%s-%s" % (et, mat)


def sos(kind, fc, order):
    k = (kind, fc if np.isscalar(fc) else tuple(fc), order)
    if k not in _SOS:
        _SOS[k] = butter(order, fc, kind, fs=SR, output="sos")
    return _SOS[k]


def read_any(path):
    sr, x = wavfile.read(path)
    if x.dtype == np.int16:
        x = x.astype(np.float64) / 32767.0
    elif x.dtype == np.int32:
        x = x.astype(np.float64) / 2147483647.0
    elif x.dtype == np.uint8:
        x = (x.astype(np.float64) - 128.0) / 127.0
    else:
        x = x.astype(np.float64)
    return to_mono(x), int(sr)


def to_mono(x):
    x = np.asarray(x, np.float64)
    if x.ndim == 2:
        x = x.mean(axis=1) if x.shape[1] <= 8 else x.mean(axis=0)
    return x


def resample(x, sr):
    if int(sr) == SR:
        return np.asarray(x, np.float64)
    g = math.gcd(SR, int(sr))
    return resample_poly(np.asarray(x, np.float64), SR // g, int(sr) // g)


def recut(x, win, pre=0.03):
    n = int(round(float(win) * SR))
    x = np.asarray(x, np.float64)
    if len(x) < 2 * n:
        x = np.pad(x, (0, 2 * n - len(x)))
    return np.asarray(_extract(x.astype(np.float32), SR, float(win), pre), np.float64)


def gate(x):
    y = np.array(x, np.float64, copy=True)
    e, f = int(GATE_END_S * SR), int(GATE_FADE_S * SR)
    lo = e - f
    if len(y) <= lo:
        return y
    seg = y[lo:e]
    seg *= np.linspace(1.0, 0.0, f)[:len(seg)]
    y[e:] = 0.0
    return y


def finish(x, foot):
    x = np.asarray(x, np.float64)
    dcv = float(x.mean()) if len(x) else 0.0
    x = x - dcv
    if len(x) > 30:
        x = sosfiltfilt(sos("highpass", 10.0, 2), x)
    x = np.asarray(normalize_loudness(x.astype(np.float32), target_rms=target_for("normal")), np.float64)
    x = np.round(np.clip(x, -1.0, 1.0) * 32767.0) / 32767.0
    a = sosfiltfilt(sos("lowpass", 16000.0, 8), x) if len(x) > 60 else x.copy()
    if foot:
        return {"x5": x, "desc": gate(x), "ana": gate(a), "dc": dcv}
    return {"x5": x, "desc": x, "ana": a, "dc": dcv}


def chain(x, sr=SR, win=None, pre=0.03, foot=False):
    x = resample(to_mono(x), sr)
    if win is not None:
        x = recut(x, win, pre)
    return finish(x, foot)


def bp(x, lo, hi, order=4):
    return sosfiltfilt(sos("bandpass", [lo, hi], order), x)


def d1_comb(x, g):
    a = np.zeros(257)
    a[0], a[256] = 1.0, -float(g)
    return lfilter([1.0], a, x)


def allpass_smear(x, tau_ms):
    n = len(x)
    m = 1 << int(math.ceil(math.log2(n + int(0.02 * SR) + 1)))
    f = np.fft.rfftfreq(m, 1.0 / SR)
    tau = tau_ms / 1000.0
    d = np.clip((f - 2000.0) / 14000.0, 0.0, 1.0) * tau
    ph = -2.0 * np.pi * np.concatenate([[0.0], np.cumsum((d[1:] + d[:-1]) * 0.5 * np.diff(f))])
    y = np.fft.irfft(np.fft.rfft(x, m) * np.exp(1j * ph), m)
    return y[:n]


def d2_smear(x, tau_ms):
    lo = sosfiltfilt(sos("lowpass", 2000.0, 4), x)
    hi = sosfiltfilt(sos("highpass", 2000.0, 4), x)
    return lo + allpass_smear(hi, tau_ms)


def d3_lowpass(x, fc=12000.0):
    return sosfiltfilt(sos("lowpass", fc, 8), x)


def noise_like(n, j):
    a, hp = HISS_SPECTRA[j % len(HISS_SPECTRA)]
    z = np.asarray(hiss(5000 + j, a, hp, dur=max(n / SR, 0.1) + 0.01), np.float64)
    return z[:n] if len(z) >= n else np.pad(z, (0, n - len(z)))


def d5_hiss(x, snr_db, j):
    z = noise_like(len(x), j)
    ps = float(np.mean(bp(x, 2000.0, 16000.0) ** 2))
    pn = float(np.mean(bp(z, 2000.0, 16000.0) ** 2))
    if ps <= 0 or pn <= 0:
        return x + z * 1e-4
    return x + z * math.sqrt(ps / (pn * 10 ** (snr_db / 10.0)))


def d8_truncate(x, end_s=0.5, fade_s=0.03):
    y = np.array(x, np.float64, copy=True)
    e, f = int(end_s * SR), int(fade_s * SR)
    y[e - f:e] *= np.linspace(1.0, 0.0, f)
    y[e:] = 0.0
    return y


def hard_clip(x, frac):
    t = float(np.quantile(np.abs(x), 1.0 - frac))
    return np.clip(x, -t, t) if t > 0 else x


def pink(n, seed):
    s = np.fft.rfft(np.random.default_rng(seed).standard_normal(n))
    f = np.fft.rfftfreq(n, 1.0 / SR)
    g = np.zeros_like(f)
    g[1:] = np.maximum(f[1:], 20.0) ** -0.5
    return np.fft.irfft(s * g, n)


def flaw(x, kind, seed):
    if kind == "clean":
        return x
    if kind == "clip02":
        return hard_clip(x, 0.002)
    if kind == "clip1":
        return hard_clip(x, 0.01)
    if kind == "pink40":
        z = pink(len(x), seed)
        r = math.sqrt(float(np.mean(x ** 2)))
        rz = math.sqrt(float(np.mean(z ** 2)))
        return x + z * (r * 10 ** (-40.0 / 20.0) / max(rz, 1e-12))
    if kind == "lp11":
        return d3_lowpass(x, 11000.0)
    raise SystemExit("unknown flaw %s" % kind)


def spaced_by_source(items, n, cluster=lambda r: r["cluster"]):
    by = defaultdict(list)
    for r in items:
        by[cluster(r)].append(r)
    order = []
    rank = 0
    srcs = sorted(by)
    while len(order) < len(items):
        for s in srcs:
            if rank < len(by[s]):
                order.append(by[s][rank])
        rank += 1
    return order[:n]


def man_load(rel_path):
    with open(os.path.join(ROOT, rel_path), encoding="utf-8") as fh:
        return json.load(fh)


def ns_refs(manifest):
    return argparse.Namespace(manifest=manifest)


def foley_clusters():
    out = {}
    p = os.path.join(FOLEY, "metadata", "foleyset_metadata.csv")
    with open(p, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out["fs_%s" % r["name"][:-4]] = "fs:%s" % r["username"]
    return out


def c3gd_clusters():
    out = {}
    with open(os.path.join(C3GD, "metadata.csv"), encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out["c3gd_%s" % r["datum_id"]] = "c3gd:%s" % r["platform_id"]
    return out


class Spec:
    __slots__ = ("uid", "kind", "key", "pool", "cluster", "win", "foot", "fam", "wet", "meta", "make", "loso")

    def __init__(self, uid, kind, key, pool, cluster, win, foot, fam, wet, meta, make, loso=None):
        self.uid, self.kind, self.key, self.pool, self.cluster = uid, kind, key, pool, cluster
        self.win, self.foot, self.fam, self.wet, self.meta, self.make, self.loso = win, foot, fam, wet, meta, make, loso


def pool_flags(pool):
    fam = POOLS[pool]["family"]
    return fam, fam == "footstep", fam in ("glass", "water")


def key_flags(key):
    fam = KEYS[key]["family"]
    return fam, fam == "footstep", fam in ("glass", "water")


def heldout_specs(pool, man, sets, meta):
    spec = POOLS[pool]
    fam, foot, wet = pool_flags(pool)
    recs = {c["id"]: c for c in man["clips"]}
    out = []
    for s in spec["sets"]:
        name = "ref:%s@0.6" % s
        if name not in sets:
            continue
        for x, cid, cl in zip(sets[name], meta[name]["ids"], meta[name]["clusters"]):
            r = recs[cid]
            if spec.get("platforms") and r.get("platform") not in spec["platforms"]:
                continue
            m = {"id": cid, "set": s, "clipped_frac": float(r.get("clipped_frac", 0.0) or 0.0),
                 "platform": r.get("platform"), "caliber": r.get("caliber"), "orig_s": r.get("orig_s"),
                 "peak_s": r.get("peak_s"), "src": r.get("src"), "file15": r["files"].get("1.5"),
                 "dataset": r.get("dataset")}
            out.append(Spec("real:%s:%s" % (pool, cid), "real", None, pool, cl, "0.6", foot, fam, wet, m,
                            (lambda x=x, foot=foot: chain(x, SR, None, foot=foot))))
    return out


_IDX = {}


def eval_index():
    if "idx" not in _IDX:
        with open(os.path.join(CACHE_V1, "index.json"), encoding="utf-8") as fh:
            _IDX["idx"] = json.load(fh)
    return _IDX["idx"]


_MM = {}


def eval_audio():
    if "eval" not in _MM:
        _MM["eval"] = np.memmap(os.path.join(CACHE_V1, "eval.f32"), dtype=np.float32, mode="r")
    return _MM["eval"]


def crop_segments(entry):
    ons = sorted(int(o) for o in entry["onsets"])
    n = int(entry["n"])
    out = []
    for i, o in enumerate(ons):
        prev = ons[i - 1] if i > 0 else None
        nxt = ons[i + 1] if i + 1 < len(ons) else None
        if prev is not None and o - prev < CROP_PREV_S * SR:
            continue
        if nxt is not None and nxt - o < CROP_NEXT_S * SR:
            continue
        s = o - int(CROP_PRE_S * SR)
        if prev is not None:
            s = max(s, prev + int(CROP_PREV_S * SR))
        s = max(0, s)
        e = (nxt - int(CROP_CUT_S * SR)) if nxt is not None else n
        if e - o < int(GATE_END_S * SR):
            continue
        out.append((o, s, e))
    return ve.spaced(out, CROP_MAX)


def crop_audio(entry, s, e):
    mm = eval_audio()
    off = int(entry["off"])
    seg = np.asarray(mm[off + s: off + e], np.float64)
    return seg


def footstep_specs(pool):
    spec = POOLS[pool]
    fam, foot, wet = pool_flags(pool)
    idx = eval_index()
    out = []
    for entry in idx["eval"]:
        if entry["bucket"] != spec["bucket"]:
            continue
        for j, (o, s, e) in enumerate(crop_segments(entry)):
            ov = sum(max(0, min(e, int(b)) - max(s, int(a))) for a, b in (entry.get("clip_runs") or []))
            m = {"id": "%s#%d" % (entry["id"], j), "entry": entry["id"], "onset": o, "seg": [s, e],
                 "clipped_frac": float(ov) / max(1, e - s), "sub": entry.get("sub"),
                 "half": entry.get("half"), "dataset": entry.get("dataset")}

            def make(entry=entry, s=s, e=e, pre=0.03):
                return chain(crop_audio(entry, s, e), SR, 0.6, pre=pre, foot=True)
            out.append(Spec("real:%s:%s" % (pool, m["id"]), "real", None, pool, entry["cluster"], "0.6", True, fam,
                            wet, m, make))
    return out


def refs_file_specs(pool):
    spec = POOLS[pool]
    fam, foot, wet = pool_flags(pool)
    man = man_load(REFS_ALL)
    out = []
    for r in man["clips"]:
        if r["set"] not in spec["sets"]:
            continue
        p = os.path.join(ROOT, *r["files"]["0.6"].split("/"))

        def make(p=p, foot=foot):
            x, sr = read_any(p)
            return chain(x, sr, None, foot=foot)
        m = {"id": r["id"], "set": r["set"], "clipped_frac": float(r.get("clipped_frac", 0.0) or 0.0),
             "src": r.get("src"), "orig_s": r.get("orig_s"), "peak_s": r.get("peak_s"),
             "file15": r["files"].get("1.5"), "dataset": r.get("dataset")}
        out.append(Spec("real:%s:%s" % (pool, r["id"]), "real", None, pool, r["cluster"], "0.6", foot, fam, wet, m,
                        make))
    return out, man


def w1_real_specs(pool, specs):
    fam, foot, wet = pool_flags(pool)
    out = []
    for s in specs:
        f15 = s.meta.get("file15")
        if not f15:
            continue
        p = os.path.join(ROOT, *f15.split("/"))

        def make(p=p):
            x, sr = read_any(p)
            return chain(x, sr, 1.0)
        out.append(Spec("w1:%s" % s.uid, "real", None, pool, s.cluster, "1.0", False, fam, wet, dict(s.meta), make))
    return out


def j9_available(specs):
    ok = [s for s in specs if s.meta.get("orig_s") is not None and s.meta.get("peak_s") is not None]
    if not specs or len(ok) < len(specs):
        return False, 0.0
    fr = float(np.mean([(s.meta["orig_s"] - s.meta["peak_s"]) >= J9_MIN_AFTER_S for s in ok]))
    return fr >= J9_AVAIL, fr


def source_recut(spec, pre, win=0.6):
    if spec.pool in POOLS and POOLS[spec.pool]["src"] == "cache_v1":
        entry = next(e for e in eval_index()["eval"] if e["id"] == spec.meta["entry"])
        s, e = spec.meta["seg"]
        return chain(crop_audio(entry, s, e), SR, win, pre=pre, foot=True)
    src = spec.meta.get("src")
    x, sr = read_any(src)
    x = resample(x, sr)
    return chain(x, SR, win, pre=pre, foot=spec.foot)


def nuisance_make(spec, kind):
    def make():
        if kind in ("pre25", "pre35"):
            return source_recut(spec, 0.025 if kind == "pre25" else 0.035, float(spec.win))
        base = spec.make()
        x5 = base["x5"]
        if kind == "rt48":
            y = resample_poly(resample_poly(x5, 160, 147), 147, 160)[:len(x5)]
        elif kind == "polarity":
            y = -x5
        elif kind == "gain05":
            y = x5 * 10 ** (0.5 / 20.0)
        else:
            raise SystemExit("unknown nuisance %s" % kind)
        return finish(y, spec.foot)
    return make


def damage_make(spec, kind, dose, j):
    def make():
        base = spec.make()
        x5 = base["x5"]
        if kind == "clean":
            y = x5
        elif kind == "d1":
            y = d1_comb(x5, D1_GAINS[dose - 1])
        elif kind == "d2":
            y = d2_smear(x5, D2_TAUS[dose - 1])
        elif kind == "d3":
            y = d3_lowpass(x5)
        elif kind == "d5":
            y = d5_hiss(x5, D5_SNRS[dose - 1], j)
        elif kind == "d25":
            y = d5_hiss(d2_smear(x5, 3.0), 20.0, j)
        elif kind == "d8":
            y = d8_truncate(x5)
        else:
            raise SystemExit("unknown damage %s" % kind)
        return finish(y, spec.foot)
    return make


def flaw_make(spec, kind, seed):
    def make():
        base = spec.make()
        return finish(flaw(base["x5"], kind, seed), spec.foot)
    return make


def d6_make(spec, shift_ms, gain_db):
    def make():
        base = spec.make()
        x5 = base["x5"]
        k = int(round(shift_ms * SR / 1000.0))
        y = np.zeros_like(x5)
        if k >= 0:
            y[k:] = x5[:len(x5) - k]
        else:
            y[:k] = x5[-k:]
        return finish(y * 10 ** (gain_db / 20.0), spec.foot)
    return make


def gen_specs(route_args, man):
    have = {(et, mat) for et, mat, _t in ve.KEYS}
    for et, mat, t in EXTRA_TYPES:
        if (et, mat) not in have:
            ve.KEYS.append((et, mat, t))
    cfgs, sets, meta, _runs = ve.load_routes(route_args, man)
    out = []
    for g, m in meta.items():
        key = m["key"]
        if key not in KEYS:
            continue
        fam, foot, wet = key_flags(key)
        for v, (x, cl) in enumerate(zip(sets[g], m["clusters"])):
            mm = {"route": m["route"], "seed": int(m["seeds"][v]), "path": m["paths"][v], "prompt": m.get("prompt")}
            out.append(Spec("gen:%s:%s:%02d" % (m["route"], key, mm["seed"]), "gen", key, KEYS[key]["pool"],
                            "seed:%02d" % mm["seed"], "0.6", foot, fam, wet, mm,
                            (lambda x=x, foot=foot: chain(x, SR, None, foot=foot))))
    return cfgs, out


def raw_specs(route, raw_dir, seeds, keys):
    gj = os.path.join(raw_dir, "gen.json")
    if not os.path.isfile(gj):
        return []
    with open(gj, encoding="utf-8") as fh:
        sr = int(json.load(fh)["sr"])
    out = []
    for key in keys:
        fam, foot, wet = key_flags(key)
        for v in range(seeds):
            p = os.path.join(raw_dir, "%s_%02d.npy" % (fname(key), v))
            if not os.path.isfile(p):
                continue

            def make(p=p, sr=sr):
                return chain(np.load(p), sr, 1.0)
            out.append(Spec("w1:gen:%s:%s:%02d" % (route, key, v), "gen", key, KEYS[key]["pool"], "seed:%02d" % v,
                            "1.0", False, fam, wet, {"route": route, "seed": v, "path": ve.rel(p)}, make))
    return out


def control_specs(key_or_pool, fam, foot, wet, win):
    out = []
    sets, meta = {}, {}
    if win == "0.6":
        ve.build_hiss(sets, meta)
        for kind in ("hiss", "silent"):
            for j, x in enumerate(sets["%s@0.6" % kind]):
                out.append(Spec("ctl:%s:%s:%02d" % (key_or_pool, kind, j), "ctl", None, None, "%s:%02d" % (kind, j),
                                "0.6", foot, fam, wet, {"control": kind},
                                (lambda x=x, foot=foot: chain(x, SR, None, foot=foot))))
        return out
    spec = [(a, hp) for a, hp in HISS_SPECTRA for _ in range(HISS_PER)]
    n = int(SR * 1.0)
    for j, (a, hp) in enumerate(spec):
        x = np.asarray(hiss(1000 + j, a, hp), np.float64)
        out.append(Spec("ctl1:%s:hiss:%02d" % (key_or_pool, j), "ctl", None, None, "hiss:%02d" % j, "1.0", False, fam,
                        wet, {"control": "hiss"}, (lambda x=x: chain(x, SR, 1.0))))
    rng = np.random.default_rng(77)
    for j in range(20):
        x = np.zeros(n) if j < 10 else rng.standard_normal(n) * 10 ** (ve.SILENT_DB[j - 10] / 20.0)
        out.append(Spec("ctl1:%s:silent:%02d" % (key_or_pool, j), "ctl", None, None, "silent:%02d" % j, "1.0", False,
                        fam, wet, {"control": "silent"}, (lambda x=x: chain(x, SR, None))))
    return out


def pcm_md5(p):
    sr, x = wavfile.read(p)
    return hashlib.md5(np.ascontiguousarray(x).tobytes()).hexdigest()


def d4_specs(pool_keys):
    with open(os.path.join(ANCHOR, "route.json"), encoding="utf-8") as fh:
        info = json.load(fh)
    anchors = info.get("anchors", {})
    fsc, c3c = foley_clusters(), c3gd_clusters()
    seen, out, pairs = set(), [], []
    for key in pool_keys:
        et, mat = key.split("|")
        ids = anchors.get(key)
        if not ids:
            continue
        fam, foot, wet = key_flags(key)
        for v, aid in enumerate(ids):
            p = os.path.join(ANCHOR, "%s-%s_%02d.wav" % (et, mat, v))
            if not os.path.isfile(p):
                continue
            h = pcm_md5(p)
            if h in seen:
                continue
            seen.add(h)
            cl = fsc.get(aid) or c3c.get(aid) or "anchor:%s" % aid
            pid = "%s:%s" % (key, aid)
            for dec, d in (("raw", ANCHOR), ("same_l", os.path.join(ROUNDTRIP, "same_l")),
                           ("same_s", os.path.join(ROUNDTRIP, "same_s"))):
                q = os.path.join(d, os.path.basename(p))
                if not os.path.isfile(q):
                    continue

                def make(q=q, foot=foot):
                    x, sr = read_any(q)
                    return chain(x, sr, None, foot=foot)
                out.append(Spec("d4:%s:%s" % (dec, pid), "d4", key, KEYS[key]["pool"], cl, "0.6", foot, fam, wet,
                                {"pair": pid, "decoder": dec, "anchor": aid, "file": ve.rel(q)}, make))
            pairs.append(pid)
    return out, pairs


def route_drop(cfg):
    info = (cfg or {}).get("info") or {}
    return sorted(set(info.get("clusters", []) or []))


def decoder_named(cfg):
    info = (cfg or {}).get("info") or {}
    blob = json.dumps(info).lower()
    return bool(re.search(r"(decoder|dec_ckpt|dec_path|finetuned_decoder)[^,]*\.safetensors", blob)) or \
        "decoder_ft" in blob

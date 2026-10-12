import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
BENCH = os.path.join(SCRIPTS, "bench")
for _p in (BENCH, SCRIPTS, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _bootstrap

import atexit
import contextlib
import ctypes
import hashlib
import json
import math
import re
import socket
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from safetensors.torch import load_file, save_file
from scipy.signal import butter, sosfilt

ROOT = os.path.dirname(SCRIPTS)
SR = 44100
DS = 4096
PATCH = 256
FRAMES = 14
SHORT_FRAMES = 13
SEG = FRAMES * DS
SHORT_SEG = SHORT_FRAMES * DS
DEC_KEYS = 120
SNAP_ID = "ae12755283df9d62ca39a9b050a39a0b607b8c20"
SNAP_S = os.path.join(r"D:\hf_cache\hub\models--stabilityai--stable-audio-3-small-sfx\snapshots", SNAP_ID)
SNAP_S_ENGINE = os.path.join(r"F:\AI\cache\hf\hub\models--stabilityai--stable-audio-3-small-sfx\snapshots", SNAP_ID)
SNAP_L = r"F:\AI\cache\hf\hub\models--stabilityai--SAME-L\snapshots\54b0fe9c9fc27aa1d5eac36402788efea15f73eb"
OUT = os.path.join(ROOT, "data", "decoder_ft")
MANIFEST = os.path.join(OUT, "manifest_v1.json")
CACHE = os.path.join(OUT, "cache_v1")
GEN = os.path.join(OUT, "gen_latents_v1")
RUNS = os.path.join(OUT, "runs")
EVAL = os.path.join(OUT, "eval")
HELDOUT_JSON = os.path.join(BENCH, "heldout_sources.json")
REFS_HELDOUT = os.path.join(ROOT, "data", "refs_heldout", "manifest.json")
SIDECAR_PORT = 8765

BUCKETS = {
    "glass_ice": ["GlassBreak", "GlassDing", "IceBreak", "IceCube", "BottleCrush"],
    "impact_break": ["BoneBreak", "CanCrush", "WoodBreak", "MetalDrop", "PunchHit", "Knock", "Fall", "DoorSlam",
                     "MetalClank", "BulletDrop"],
    "click_mech": ["ButtonPress", "KeyboardTyping", "Knob", "MouseButtonClick", "Stapler", "SwitchToggle",
                   "GunMechanism", "Lock", "Latch"],
    "splash_drip": ["Splash", "Drip", "Bubble", "Pour", "LiquidShaking"],
    "metal_ring": ["Chain", "CoinJingle", "KeyJingle", "Knife", "Sword", "Pipe", "MetalScape"],
    "door_drawer": ["DoorOpenClose", "DoorSqueak", "DrawerOpenClose", "WindowOpenClose"],
    "texture": ["ClothMovement", "ClothRipping", "ClothRubbing", "Umbrella", "PaperTear", "PaperRubbing", "PenWriting",
                "PencilWriting", "PlasticRubbing", "TapePeel", "VelcroRip", "ZipperZip", "WhiteboardWriting",
                "WindowCleaning", "Magazine"],
    "human_steady": ["Breath", "Burp", "Cough", "EatDrink", "Fart", "HeartBeat", "Kiss", "Slurping", "Sneeze", "Sniffle",
                     "Snore", "StrawSip", "Faucet", "Shower", "Toilet"],
}
SUB_BUCKET = {s: b for b, ss in BUCKETS.items() for s in ss}
SHARES = {"gunshot": 7.0, "step_gravel": 4.0, "step_stone": 7.5, "step_wood": 6.5, "step_metal": 1.0, "glass_ice": 4.0,
          "impact_break": 5.0, "click_mech": 10.0, "splash_drip": 10.0, "step_other": 13.0, "metal_ring": 9.0,
          "door_drawer": 6.0, "texture": 9.0, "human_steady": 8.0}
TRANSIENT = ["gunshot", "step_gravel", "step_stone", "step_wood", "glass_ice", "impact_break", "click_mech"]
SMOOTH = ["texture", "human_steady", "splash_drip"]
LICENSES = {
    "foleyset": {"license": "CC BY 4.0", "source": "FoleySet, Zenodo record 20735877 (CC0 Freesound clips)",
                 "cite": "Wang and Lerch, FoleySet: A Multi-Level Human-Annotated Foley Sound Dataset, arXiv:2606.25980"},
    "c3gd": {"license": "CC BY 4.0", "source": "C3GD v0.2.0, Zenodo record 22286299",
             "cite": "Gurny and Quinn, Descriptor: Certus Caliber Classification Gunshot Dataset (C3GD), arXiv:2606.18135"},
    "hear": {"license": "CC0", "source": "HEAR 2021 gunshot_triangulation v1.0 full 48000, Zenodo record 5887964",
             "cite": "HEAR 2021 benchmark task gunshot_triangulation"},
}
FOOT_SURF = ["wood", "stone", "grass", "metal", "gravel"]
VOCAB_KEYS = [("shoot", "rifle"), ("shoot", "pistol"), ("reload", "rifle"), ("equip", "rifle"), ("empty_click", "pistol"),
              ("explosion", "grenade"), ("swing", "sword"), ("open", "door"), ("slam", "door"), ("break", "window"),
              ("click", "button"), ("cast", "fire_spell"), ("punch", "body"), ("horn", "truck"), ("bark", "dog")] + \
             [("footstep", s) for s in FOOT_SURF]
EVAL_KEYS = VOCAB_KEYS + [("footstep", "water"), ("splash", "water")]
ANCHOR_HELD = ["bark|dog", "horn|truck", "footstep|grass", "punch|body"]
KEY_BUCKET = {
    "shoot|rifle": "gunshot", "shoot|pistol": "gunshot", "reload|rifle": "click_mech", "equip|rifle": "click_mech",
    "empty_click|pistol": "click_mech", "explosion|grenade": "impact_break", "swing|sword": "metal_ring",
    "open|door": "door_drawer", "slam|door": "impact_break", "break|window": "glass_ice", "click|button": "click_mech",
    "cast|fire_spell": "texture", "punch|body": "impact_break", "horn|truck": None, "bark|dog": None,
    "footstep|wood": "step_wood", "footstep|stone": "step_stone", "footstep|grass": "step_other",
    "footstep|metal": "step_metal", "footstep|gravel": "step_gravel", "footstep|water": "splash_drip",
    "splash|water": "splash_drip",
}
TRANSIENT_KEYS = {"shoot|rifle": "gunshot_rifle", "shoot|pistol": "gunshot_pistol", "footstep|gravel": "step_gravel",
                  "footstep|stone": "step_stone", "footstep|wood": "step_wood", "break|window": "glass_ice"}
CLUSTER_RE = re.compile(r"^(fs|c3gd):")
FFTS = (2048, 1024, 512, 256, 128, 64, 32)
HF_FFTS = (512, 256, 128, 64, 32)
BANDS_HF = ((4000.0, 8000.0), (8000.0, 16000.0), (16000.0, None))
LSD_BANDS = ((0.0, 1000.0), (1000.0, 4000.0), (4000.0, 8000.0), (8000.0, 16000.0), (16000.0, 22050.0))
FRAME_50 = int(0.05 * SR)
COMB_HZ = SR / PATCH
FLUTTER_HZ = SR / DS / 2.0


def at_root(p):
    if p is None:
        return None
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def norm(p):
    return os.path.normcase(os.path.abspath(p))


def mix_of(b):
    return "gunshot" if b.startswith("gunshot_") else b


def md5_text(s):
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def file_md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def half_of(cluster):
    return "A" if int(md5_text(cluster), 16) % 2 == 0 else "B"


def load_json(p):
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def dump_json(obj, p):
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, default=_jsonable)
    os.replace(tmp, p)


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, set):
        return sorted(o)
    return str(o)


def load_ae(device, snap=SNAP_S):
    from stable_audio_3.loading_utils import load_autoencoder
    ae = load_autoencoder(os.path.join(snap, "model_config.json"), os.path.join(snap, "model.safetensors"), device)
    return ae.float().eval().requires_grad_(False)


def fresh_decoder(device, snap=SNAP_S):
    ae = load_ae(device, snap)
    dec = ae.decoder
    del ae
    return dec.float()


def unpatch(y):
    return rearrange(y, "b (c h) l -> b c (l h)", h=PATCH)


def decode_train(ae, dec, z):
    return unpatch(dec(ae.bottleneck.decode(z)))


def _rng_get(t):
    return torch.cuda.get_rng_state(t.device) if t.is_cuda else torch.get_rng_state()


def _rng_set(t, st):
    if t.is_cuda:
        torch.cuda.set_rng_state(st, t.device)
    else:
        torch.set_rng_state(st)


def twin_decode(ae, dec, ref, z):
    b = ae.bottleneck.decode(z)
    st = _rng_get(b)
    y = dec(b)
    _rng_set(b, st)
    with torch.no_grad():
        r = ref(b)
    return unpatch(y), unpatch(r)


def seeded_decode(ae, dec, z, seed):
    mode = ae.bottleneck.training
    ae.bottleneck.train(False)
    try:
        with torch.random.fork_rng(devices=[z.device] if z.is_cuda else []):
            torch.manual_seed(int(seed))
            return decode_train(ae, dec, z)
    finally:
        ae.bottleneck.train(mode)


def encode(ae, x):
    return ae.encode(x)


def mid(y):
    return y.mean(1)


def side(y):
    return 0.5 * (y[:, 0] - y[:, 1])


def export_decoder(dec, path, meta):
    sd = {k: v.detach().float().contiguous().cpu() for k, v in dec.state_dict().items()}
    if len(sd) != DEC_KEYS:
        raise SystemExit("decoder state_dict has %d keys, expected %d" % (len(sd), DEC_KEYS))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    save_file(sd, tmp, metadata={"sarge": json.dumps(meta, default=_jsonable)})
    os.replace(tmp, path)


def load_decoder(dec, path):
    dec.load_state_dict(load_file(path), strict=True)
    return dec


def decoder_meta(path):
    from safetensors import safe_open
    with safe_open(path, framework="pt") as f:
        m = f.metadata() or {}
    return json.loads(m.get("sarge", "{}"))


_WINS = {}


def _win(n, device):
    k = (n, str(device))
    w = _WINS.get(k)
    if w is None:
        w = torch.hann_window(n, device=device)
        _WINS[k] = w
    return w


def stft_c(x, n, hop=None):
    return torch.stft(x.float(), n, hop_length=hop or n // 4, win_length=n, window=_win(n, x.device), center=True,
                      return_complex=True)


def stft_mag(x, n, hop=None):
    X = stft_c(x, n, hop)
    return torch.sqrt(torch.clamp(X.real.square() + X.imag.square(), min=1e-8))


def frame_weights(mask, n, hop):
    p = F.avg_pool1d(mask[:, None].float(), n, hop, n // 2, count_include_pad=False)[:, 0]
    return (p > 1.0 - 1e-6).float()


def _bins(n, fmin, fmax, sr=SR):
    nb = n // 2 + 1
    lo = 0 if not fmin or fmin <= 0 else min(nb, int(math.ceil(fmin * n / sr)))
    hi = nb if fmax is None else min(nb, int(math.ceil(fmax * n / sr)))
    return lo, hi


def mrstft(x, y, ffts=FFTS, fmin=0.0, fmax=None, frame_mask=None, den=None, sr=SR, per_res=False):
    tot = []
    for n in ffts:
        lo, hi = _bins(n, fmin, fmax, sr)
        if hi - lo < 1:
            continue
        X = stft_mag(x, n)[:, lo:hi]
        Y = stft_mag(y, n)[:, lo:hi]
        D = Y if den is None else stft_mag(den, n)[:, lo:hi]
        if frame_mask is not None:
            fw = frame_weights(frame_mask, n, n // 4)[:, None, :X.shape[-1]]
            sc = torch.linalg.vector_norm((Y - X) * fw) / torch.linalg.vector_norm(D * fw).clamp(min=1e-8)
            lm = ((torch.log(X) - torch.log(Y)).abs() * fw).sum() / (fw.sum() * (hi - lo)).clamp(min=1.0)
        else:
            sc = torch.linalg.vector_norm(Y - X) / torch.linalg.vector_norm(D).clamp(min=1e-8)
            lm = (torch.log(X) - torch.log(Y)).abs().mean()
        tot.append(sc + lm)
    if per_res:
        return torch.stack(tot)
    return torch.stack(tot).mean()


def band_env(x, n=1024, hop=256, sr=SR):
    X = stft_c(x, n, hop)
    P = X.real.square() + X.imag.square()
    out = []
    for lo, hi in BANDS_HF:
        a, b = _bins(n, lo, hi, sr)
        out.append(torch.log(P[:, a:b].sum(1) + 1e-8))
    return torch.stack(out, 1)


_SOS = {}


def _sos(kind, fc, sr=SR, order=4):
    k = (kind, fc, sr, order)
    if k not in _SOS:
        _SOS[k] = butter(order, fc, kind, fs=sr, output="sos")
    return _SOS[k]


def hp4k(x, sr=SR):
    return sosfilt(_sos("highpass", 4000.0, sr), np.asarray(x, np.float64))


def hp8k(x, sr=SR):
    return sosfilt(_sos("highpass", 8000.0, sr), np.asarray(x, np.float64))


def loudest(mono, win):
    n = mono.shape[0]
    c = np.empty(n + 1)
    c[0] = 0.0
    np.cumsum(mono.astype(np.float64) ** 2, out=c[1:])
    lo = win - 1 - (win - 1) // 2
    hi = (win - 1) // 2 + 1
    up = np.empty(n)
    m = n + 1 - hi
    up[:m] = c[hi:]
    up[m:] = c[n]
    low = np.empty(n)
    low[:lo] = 0.0
    low[lo:] = c[:n - lo]
    return int(np.argmax(up - low))


def window(mono, sr, dur, pre=0.03):
    n = int(dur * sr)
    if mono.shape[0] <= n:
        return 0, mono.shape[0]
    c = loudest(mono, max(1, int(0.005 * sr)))
    s = max(0, c - int(pre * sr))
    e = min(mono.shape[0], s + n)
    return max(0, e - n), e


def kurt(h):
    p2 = float(np.mean(h * h))
    if p2 < 1e-14:
        return float("nan")
    return float(np.mean(h ** 4) / (p2 * p2))


def frames(x, n):
    nf = x.shape[0] // n
    return x[:nf * n].reshape(nf, n)


def active_frames(h, fl=FRAME_50, db=40.0):
    f = frames(h, fl)
    if not len(f):
        return np.zeros(0, int)
    r = np.sqrt(np.mean(f * f, axis=1))
    top = float(r.max())
    if top <= 0:
        return np.zeros(0, int)
    return np.flatnonzero(r >= top * 10.0 ** (-db / 20.0))


def hfk_win(x, h=None, win=None):
    h = hp4k(x) if h is None else h
    s, e = window(np.asarray(x, np.float64), SR, 0.6, 0.03) if win is None else win
    return kurt(h[s:e])


def hfk_frame(h, idx):
    if not len(idx):
        return float("nan")
    f = frames(h, FRAME_50)[idx]
    return float(np.nanmedian([kurt(r) for r in f]))


def spike_rate(h, idx):
    if not len(idx):
        return float("nan")
    f = frames(h, FRAME_50)[idx]
    med = np.median(f, axis=1, keepdims=True)
    sig = 1.4826 * np.median(np.abs(f - med), axis=1, keepdims=True)
    cnt = np.sum(np.abs(f - med) > 4.0 * np.maximum(sig, 1e-12))
    return float(cnt / (len(idx) * 0.05))


def env_ms(h, ms=1.0, hop_ms=None):
    n = max(1, int(round(ms * SR / 1000.0)))
    hop = n if hop_ms is None else max(1, int(round(hop_ms * SR / 1000.0)))
    c = np.concatenate([[0.0], np.cumsum(np.asarray(h, np.float64) ** 2)])
    starts = np.arange(0, max(1, h.shape[0] - n + 1), hop)
    return np.sqrt(np.maximum(c[starts + n] - c[starts], 0.0) / n)


def hf_env_corr(h_ref, h_dec):
    a = env_ms(h_ref)
    b = env_ms(h_dec)
    m = min(len(a), len(b))
    a, b = a[:m], b[:m]
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def hf_floor_db(h, idx):
    if not len(idx):
        return float("nan")
    e = np.sort(np.mean(frames(h, FRAME_50)[idx] ** 2, axis=1))
    k = max(1, len(e) // 2)
    return float(10.0 * np.log10(np.mean(e[:k]) + 1e-20))


def energy_db(x):
    return float(10.0 * np.log10(np.mean(np.asarray(x, np.float64) ** 2) + 1e-20))


_HANN = {}


def np_stft_pow(x, n, hop):
    x = np.asarray(x, np.float64)
    pad = np.pad(x, (n // 2, n // 2), mode="reflect" if x.shape[0] > n // 2 else "constant")
    if n not in _HANN:
        _HANN[n] = np.hanning(n + 1)[:-1]
    fr = np.lib.stride_tricks.sliding_window_view(pad, n)[::hop]
    X = np.fft.rfft(fr * _HANN[n], axis=1)
    return (X.real ** 2 + X.imag ** 2).T


def spec_pair(ref, dec):
    return {n: (np_stft_pow(ref, n, n // 4), np_stft_pow(dec, n, n // 4)) for n in (2048, 512)}


def lsd_bands(ref, dec, n=2048, hop=512, spec=None):
    P, Q = spec[n] if spec is not None and hop == n // 4 else (np_stft_pow(ref, n, hop), np_stft_pow(dec, n, hop))
    m = min(P.shape[1], Q.shape[1])
    P, Q = P[:, :m], Q[:, :m]
    fe = P.sum(0)
    act = fe >= fe.max() * 1e-6 if fe.max() > 0 else np.zeros(m, bool)
    out = {}
    for lo, hi in LSD_BANDS:
        a, b = _bins(n, lo, hi)
        if b <= a or not act.any():
            out["%d-%d" % (lo / 1000, hi / 1000)] = float("nan")
            continue
        d = 10 * np.log10(Q[a:b, act] + 1e-10) - 10 * np.log10(P[a:b, act] + 1e-10)
        out["%d-%d" % (lo / 1000, hi / 1000)] = float(np.mean(np.sqrt(np.mean(d * d, axis=0))))
    return out


_MEL = {}


def _mel_fb(n, n_mels, fmax):
    k = (n, n_mels, fmax)
    if k not in _MEL:
        import torchaudio
        fb = torchaudio.functional.melscale_fbanks(n // 2 + 1, 0.0, float(fmax or SR / 2), n_mels, SR, norm="slaney",
                                                   mel_scale="slaney")
        _MEL[k] = fb.numpy().astype(np.float64)
    return _MEL[k]


def mel_dist(ref, dec, fmax=None, spec=None):
    tot = []
    for n, nm in ((2048, 320 if fmax is None else 64), (512, 80 if fmax is None else 16)):
        fb = _mel_fb(n, nm, fmax)
        P, Q = spec[n] if spec is not None else (np_stft_pow(ref, n, n // 4), np_stft_pow(dec, n, n // 4))
        A = np.sqrt(P).T @ fb
        B = np.sqrt(Q).T @ fb
        m = min(len(A), len(B))
        tot.append(np.mean(np.abs(np.log10(np.maximum(A[:m], 1e-5)) - np.log10(np.maximum(B[:m], 1e-5)))))
    return float(np.mean(tot))


def si_sdr(ref, est):
    r = np.asarray(ref, np.float64)
    e = np.asarray(est, np.float64)
    r = r - r.mean()
    e = e - e.mean()
    rr = float(np.dot(r, r))
    if rr < 1e-20:
        return float("nan")
    s = np.dot(e, r) / rr * r
    n = e - s
    return float(10 * np.log10((np.dot(s, s) + 1e-20) / (np.dot(n, n) + 1e-20)))


def flux_onsets(x, k=7, n=256, hop=64, rel=0.3, gap_s=0.05):
    x = np.asarray(x, np.float64)
    out = [loudest(x, max(1, int(0.005 * SR)))]
    if x.shape[0] > n:
        M = np.sqrt(np_stft_pow(x, n, hop))
        a, _b = _bins(n, 4000.0, None)
        fl = np.maximum(np.diff(M[a:], axis=1), 0.0).sum(0)
        if fl.size and fl.max() > 0:
            pk = np.flatnonzero((fl[1:-1] >= fl[:-2]) & (fl[1:-1] > fl[2:])) + 1
            pk = pk[fl[pk] >= rel * fl.max()]
            pk = pk[np.argsort(-fl[pk])]
            gap = int(gap_s * SR)
            for p in pk:
                s = int((p + 1) * hop)
                if all(abs(s - o) >= gap for o in out):
                    out.append(s)
                if len(out) >= k + 1:
                    break
    return sorted(int(min(max(o, 0), x.shape[0] - 1)) for o in out)


def attack(ref, dec, onsets):
    h_r, h_d = hp4k(ref), hp4k(dec)
    er, ed = env_ms(h_r, 1.0, 0.25), env_ms(h_d, 1.0, 0.25)
    rr, rd, cr, cd, pr, pd = [], [], [], [], [], []
    for on in onsets:
        i = int(on / (0.25e-3 * SR))
        lo, hi = max(0, i - 8), min(len(er), i + 60)
        if hi - lo < 4:
            continue
        for env, out in ((er, rr), (ed, rd)):
            seg = env[lo:hi]
            pk = lo + int(np.argmax(seg))
            top = env[pk]
            if top <= 0:
                out.append(float("nan"))
                continue
            j = pk
            while j > max(0, i - 80) and env[j] >= 0.1 * top:
                j -= 1
            t90 = j + int(np.argmax(env[j:pk + 1] >= 0.9 * top))
            out.append((t90 - j) * 0.25)
        w = int(0.01 * SR)
        a, b = max(0, on - w), min(len(ref), on + w)
        for sig, out in ((ref, cr), (dec, cd)):
            s = np.asarray(sig[a:b], np.float64)
            r = np.sqrt(np.mean(s * s)) if s.size else 0.0
            out.append(float(20 * np.log10((np.max(np.abs(s)) + 1e-12) / (r + 1e-12))) if s.size else float("nan"))
        p0, p1 = max(0, on - int(0.02 * SR)), max(0, on - int(0.005 * SR))
        for sig, out in ((ref, pr), (dec, pd)):
            s = np.asarray(sig, np.float64)
            if p1 - p0 < 16:
                out.append(float("nan"))
                continue
            e_pre = np.mean(s[p0:p1] ** 2)
            e_on = np.mean(s[on:on + w] ** 2) if on + 16 < len(s) else 0.0
            out.append(float(10 * np.log10((e_pre + 1e-20) / (e_on + 1e-20))))
    f = lambda v: float(np.nanmean(v)) if len(v) and np.isfinite(v).any() else float("nan")
    rr, rd, cr, cd, pr, pd = map(np.asarray, (rr, rd, cr, cd, pr, pd))
    return {"rise_ref_ms": f(rr), "rise_dec_ms": f(rd), "rise_err_ms": f(np.abs(rd - rr)) if len(rr) else float("nan"),
            "crest_ref_db": f(cr), "crest_dec_db": f(cd), "crest_err_db": f(np.abs(cd - cr)) if len(cr) else float("nan"),
            "pre_ref_db": f(pr), "pre_dec_db": f(pd), "onsets": int(len(onsets))}


def gap_hiss_db(ref, dec, db=50.0, ms=10.0):
    n = int(ms * SR / 1000)
    fr = frames(np.asarray(ref, np.float64), n)
    if not len(fr):
        return float("nan")
    r = np.sqrt(np.mean(fr ** 2, axis=1))
    top = r.max()
    gap = np.flatnonzero(r <= top * 10 ** (-db / 20.0)) if top > 0 else np.arange(len(r))
    if not len(gap):
        return float("nan")
    hd = frames(hp8k(dec), n)[gap]
    return float(10 * np.log10(np.mean(hd ** 2) + 1e-20))


def welch_pow(x, n=8192):
    from scipy.signal import welch
    f, p = welch(np.asarray(x, np.float64), fs=SR, nperseg=min(n, len(x)), noverlap=min(n, len(x)) // 2)
    return f, p


def comb_172(x, n=8192):
    f, p = welch_pow(x, n)
    step = COMB_HZ * n / SR
    best = -np.inf
    k = 1
    while True:
        c = int(round(k * step))
        if c + 14 >= len(p):
            break
        nb = np.concatenate([p[c - 12:c - 2], p[c + 3:c + 13]])
        best = max(best, 10 * np.log10((p[c] + 1e-30) / (np.median(nb) + 1e-30)))
        k += 1
    return float(best)


def flutter_538(x, ms=10.0):
    e = env_ms(np.asarray(x, np.float64), ms)
    if len(e) < 32:
        return float("nan")
    v = np.log(e + 1e-9)
    v = v - v.mean()
    nfft = 4096
    S = np.abs(np.fft.rfft(v * np.hanning(len(v)), nfft)) ** 2
    fr = np.fft.rfftfreq(nfft, ms / 1000.0)
    c = int(np.argmin(np.abs(fr - FLUTTER_HZ)))
    band = np.flatnonzero((fr >= 3.0) & (fr <= 8.0) & (np.abs(fr - FLUTTER_HZ) > 0.6))
    if not len(band):
        return float("nan")
    return float(10 * np.log10((S[max(0, c - 1):c + 2].max() + 1e-30) / (np.median(S[band]) + 1e-30)))


def make_chirp(rng, n):
    span = float(rng.uniform(2.0, 6.5))
    span = min(span, math.log2(20000.0 / 100.0))
    lo = 100.0 * 2.0 ** float(rng.uniform(0.0, math.log2(20000.0 / 100.0) - span))
    hi = lo * 2.0 ** span
    f0, f1 = (lo, hi) if rng.uniform() < 0.5 else (hi, lo)
    t = np.arange(n) / float(n)
    finst = f0 * (f1 / f0) ** t
    ph = 2 * np.pi * np.cumsum(finst) / SR
    lvl = 10 ** (float(rng.uniform(-24.0, -6.0)) / 20.0)
    fade = min(int(0.005 * SR), n // 4)
    g = np.ones(n)
    g[:fade] = np.linspace(0, 1, fade)
    g[-fade:] = np.linspace(1, 0, fade)
    return (lvl * np.sin(ph) * g).astype(np.float32), finst


def chirp_alias_db(y, finst, n=2048, hop=512):
    P = np_stft_pow(y, n, hop)
    fr = np.fft.rfftfreq(n, 1.0 / SR)
    tot_in, tot_out = 0.0, 0.0
    for t in range(P.shape[1]):
        s = min(len(finst) - 1, t * hop)
        if s < n or s > len(finst) - n:
            continue
        f = finst[s]
        inb = (fr >= f / 2 ** (1 / 3)) & (fr <= f * 2 ** (1 / 3))
        tot_in += P[inb, t].sum()
        tot_out += P[~inb & (fr > 50.0), t].sum()
    if tot_in <= 0:
        return float("nan")
    return float(10 * np.log10((tot_out + 1e-30) / tot_in))


def det_crop(x, n=SEG, pre_s=0.10, peak_db=-6.0):
    x = np.asarray(x, np.float32)
    c = loudest(x, max(1, int(0.005 * SR))) if x.shape[0] else 0
    s = max(0, c - int(pre_s * SR))
    seg = x[s:s + n].copy()
    if seg.shape[0] < n:
        fo = min(int(0.01 * SR), seg.shape[0] // 2)
        if fo > 0 and s + seg.shape[0] >= x.shape[0]:
            seg[-fo:] *= np.linspace(1.0, 0.0, fo, dtype=np.float32)
        seg = np.pad(seg, (0, n - seg.shape[0]))
    pk = float(np.max(np.abs(seg))) if seg.size else 0.0
    if pk > 0:
        seg *= np.float32(10 ** (peak_db / 20.0) / pk)
    return np.stack([seg, seg]), s


def clip_pair_metrics(ref, dec, onsets=None):
    ref = np.asarray(ref, np.float64)
    dec = np.asarray(dec, np.float64)
    h_r, h_d = hp4k(ref), hp4k(dec)
    win = window(ref, SR, 0.6, 0.03)
    idx = active_frames(h_r)
    sp = spec_pair(ref, dec)
    out = {"hfk_win_ref": kurt(h_r[win[0]:win[1]]), "hfk_win": kurt(h_d[win[0]:win[1]]),
           "hfk_frame_ref": hfk_frame(h_r, idx), "hfk_frame": hfk_frame(h_d, idx),
           "spike_rate_ref": spike_rate(h_r, idx), "spike_rate": spike_rate(h_d, idx),
           "hf_env_corr": hf_env_corr(h_r, h_d), "hf_floor_db_ref": hf_floor_db(h_r, idx),
           "hf_floor_db": hf_floor_db(h_d, idx), "hf_energy_db_ref": energy_db(h_r), "hf_energy_db": energy_db(h_d),
           "mel_dist": mel_dist(ref, dec, spec=sp), "mel2k": mel_dist(ref, dec, 2000.0, spec=sp),
           "si_sdr": si_sdr(ref, dec), "gap_hiss_db": gap_hiss_db(ref, dec)}
    for k, v in lsd_bands(ref, dec, spec=sp).items():
        out["lsd_" + k] = v
    out.update(attack(ref, dec, flux_onsets(ref) if onsets is None else onsets))
    return out


def self_metrics(x):
    x = np.asarray(x, np.float64)
    h = hp4k(x)
    idx = active_frames(h)
    return {"hfk_win": hfk_win(x, h), "hfk_frame": hfk_frame(h, idx), "spike_rate": spike_rate(h, idx),
            "hf_energy_db": energy_db(h), "hf_floor_db": hf_floor_db(h, idx)}


def log_ratio(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(a / b)


def _alive(pid):
    import ctypes.wintypes as w
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.OpenProcess.argtypes = [w.DWORD, w.BOOL, w.DWORD]
    k.OpenProcess.restype = w.HANDLE
    k.GetExitCodeProcess.argtypes = [w.HANDLE, ctypes.POINTER(w.DWORD)]
    k.GetExitCodeProcess.restype = w.BOOL
    k.CloseHandle.argtypes = [w.HANDLE]
    h = k.OpenProcess(0x1000, False, int(pid))
    if not h:
        return ctypes.get_last_error() == 5
    try:
        code = w.DWORD()
        if not k.GetExitCodeProcess(h, ctypes.byref(code)):
            return True
        return code.value == 259
    finally:
        k.CloseHandle(h)


def sidecar_up(port=SIDECAR_PORT):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.5)
    try:
        return s.connect_ex(("127.0.0.1", int(port))) == 0
    finally:
        s.close()


def lock_holder():
    p = os.path.join(OUT, "gpu.lock")
    if not os.path.isfile(p):
        return None
    try:
        info = load_json(p)
    except Exception:
        return None
    pid = int(info.get("pid", 0) or 0)
    if pid and pid != os.getpid() and _alive(pid):
        return info
    return None


@contextlib.contextmanager
def gpu_guard(tag):
    if sidecar_up():
        raise SystemExit("gpu_guard(%s): the sidecar is listening on 127.0.0.1:%d; stop it first" % (tag, SIDECAR_PORT))
    held = lock_holder()
    if held:
        raise SystemExit("gpu_guard(%s): %s (pid %s) holds %s" % (tag, held.get("tag"), held.get("pid"),
                                                                  os.path.join(OUT, "gpu.lock")))
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "gpu.lock")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump({"pid": os.getpid(), "tag": tag, "started": time.strftime("%Y-%m-%d %H:%M:%S")}, fh)

    def release():
        try:
            if os.path.isfile(p) and int(load_json(p).get("pid", 0)) == os.getpid():
                os.remove(p)
        except Exception:
            pass

    atexit.register(release)
    try:
        yield
    finally:
        release()


def held_sources(hs):
    ups = hs["foleyset"]["uploaders"]
    guns = hs["c3gd"]["guns"]
    users = {u for u, v in ups.items() if v["pool"] != "train"} | set(hs["foleyset"]["heldout_uploaders"])
    gs = {g for g, v in guns.items() if v["pool"] != "train"} | set(hs["c3gd"]["heldout_guns"])
    return users, gs


def audit(man):
    meta = man["meta"]
    if file_md5(HELDOUT_JSON) != meta["heldout_sources_md5"]:
        raise SystemExit("audit: heldout_sources.json changed since the manifest was built; rebuild the manifest")
    hs = load_json(HELDOUT_JSON)
    held_users, held_guns = held_sources(hs)
    esc_users = set(meta["esc_users_dropped"])
    block = set(meta["block_md5"])
    ref = load_json(REFS_HELDOUT)
    ref_paths = {norm(c["src"]) for c in ref["clips"]}
    ref_md5 = set(meta["refs_heldout_md5"])
    forbid = [norm(os.path.join(ROOT, "data", d)) for d in ("esc50", "real_footsteps", "refs")]
    tv = [r for r in man["records"] if r["set"] in ("train", "val")]
    ev = [r for r in man["records"] if r["set"] == "eval"]
    bad = defaultdict(list)
    for r in tv:
        c = r["cluster"]
        p = norm(r["path"])
        if not CLUSTER_RE.match(c):
            bad["cluster_format"].append(r["id"])
        if c.startswith("fs:") and (c[3:] in held_users or c[3:] in esc_users):
            bad["held_or_esc_uploader"].append(r["id"])
        if c.startswith("c3gd:") and c[5:] in held_guns:
            bad["held_gun"].append(r["id"])
        if r["md5"] in block:
            bad["md5_in_block"].append(r["id"])
        if p in ref_paths or r["md5"] in ref_md5:
            bad["in_refs_heldout"].append(r["id"])
        if r["dataset"] not in ("foleyset", "c3gd") or r["license"] != "CC BY 4.0":
            bad["dataset_or_license"].append(r["id"])
        if any(p == f or p.startswith(f + os.sep) for f in forbid) or "sonniss" in p.lower():
            bad["forbidden_path"].append(r["id"])
    if any(r["dataset"] == "hear" for r in tv):
        bad["hear_outside_eval"].append("hear")
    tv_md5 = {r["md5"] for r in tv}
    tv_cl = {r["cluster"] for r in tv}
    for r in ev:
        if r["md5"] in tv_md5:
            bad["eval_md5_in_train_val"].append(r["id"])
        if r["cluster"] in tv_cl:
            bad["eval_cluster_in_train_val"].append(r["id"])
    if bad:
        raise SystemExit("audit FAILED: %s" % json.dumps({k: [len(v), v[:10]] for k, v in bad.items()}))
    return {"train": sum(r["set"] == "train" for r in tv), "val": sum(r["set"] == "val" for r in tv), "eval": len(ev),
            "train_val_clusters": len(tv_cl), "held_users": len(held_users), "held_guns": len(held_guns),
            "esc_users_dropped": len(esc_users), "block_md5": len(block), "refs_heldout_sources": len(ref_paths),
            "checks": ["cluster format ^(fs|c3gd):", "no held-out uploader, gun or ESC uploader",
                       "no md5 in BLOCK", "no refs_heldout path or md5", "FoleySet/C3GD only, CC BY 4.0",
                       "no esc50/real_footsteps/refs/sonniss path", "HEAR only in eval",
                       "eval md5 and clusters disjoint from train/val"]}


def capped_probs(w, caps):
    w = np.asarray(w, np.float64)
    caps = np.asarray(caps, np.float64)
    p = w / w.sum()
    fixed = np.zeros(len(p), bool)
    for _ in range(len(p) + 1):
        over = (~fixed) & (p > caps + 1e-12)
        if not over.any():
            break
        fixed |= over
        p[fixed] = caps[fixed]
        rest = 1.0 - p[fixed].sum()
        free = ~fixed
        if not free.any() or rest <= 0:
            break
        p[free] = w[free] / w[free].sum() * rest
    return p / p.sum()


PLAN_STEPS = 12000
PLAN_REAL = 15
MAX_PASSES = 80.0


def coverage(act, crop_s=SEG / SR):
    return np.minimum(1.0, crop_s / np.maximum(np.asarray(act, np.float64), 1e-3))


def project_caps(w, sj, scap, ccap, iters=200):
    w = np.asarray(w, np.float64)
    p = w / w.sum()
    ns = len(scap)
    for _ in range(iters):
        p = np.minimum(p, ccap)
        ps = np.bincount(sj, weights=p, minlength=ns)
        f = np.minimum(1.0, scap / np.maximum(ps, 1e-30))
        p = p * f[sj]
        tot = p.sum()
        if 1.0 - tot < 1e-10:
            break
        ps = np.bincount(sj, weights=p, minlength=ns)
        free = (p < ccap * (1 - 1e-9)) & (ps[sj] < scap[sj] * (1 - 1e-9))
        if not free.any():
            break
        p[free] += (1.0 - tot) * w[free] / w[free].sum()
    return p / p.sum()


def bucket_tables(recs, steps=PLAN_STEPS, n_real=PLAN_REAL, max_passes=MAX_PASSES):
    by = defaultdict(list)
    for i, r in enumerate(recs):
        by[mix_of(r["bucket"])].append(i)
    tot_share = sum(SHARES.get(m, 0.0) for m in by)
    out = {}
    for m, idx in sorted(by.items()):
        idx = np.array(idx)
        w = np.array([max(recs[i]["weight"], 1e-6) for i in idx])
        act = np.array([max(recs[i]["active_s"], 1e-3) for i in idx])
        src = [recs[i]["cluster"] for i in idx]
        names = sorted(set(src))
        pos = {s: j for j, s in enumerate(names)}
        sj = np.array([pos[s] for s in src])
        sa = np.bincount(sj, weights=act, minlength=len(names))
        caps = np.maximum.reduce([np.full(len(names), 0.10), np.full(len(names), 1.5 / len(names)),
                                  0.5 * sa / sa.sum()])
        draws = steps * n_real * SHARES.get(m, 0.0) / max(tot_share, 1e-9)
        ccap = max_passes / np.maximum(draws * coverage(act), 1e-9) if max_passes and draws > 0 else np.full(len(idx), 1.0)
        if ccap.sum() < 1.0:
            ccap = ccap / ccap.sum()
        p = project_caps(w, sj, caps, ccap)
        ps = np.bincount(sj, weights=p, minlength=len(names))
        out[m] = {"idx": idx, "p": p, "sources": names, "source_p": ps, "caps": caps, "clip_caps": ccap}
    return out


def expected_passes(recs, steps=PLAN_STEPS, n_real=PLAN_REAL):
    tabs = bucket_tables(recs)
    out = {}
    tot = sum(SHARES.get(m, 0.0) for m in tabs)
    for m, t in tabs.items():
        sh = SHARES.get(m, 0.0) / tot
        draws = steps * n_real * sh * t["p"]
        act = np.array([max(recs[i]["active_s"], 1e-3) for i in t["idx"]])
        passes = draws * coverage(act)
        out[m] = {"clips": int(len(t["idx"])), "sources": len(t["sources"]), "share": round(sh * 100, 2),
                  "draws_mean": round(float(draws.mean()), 1), "passes_med": round(float(np.median(passes)), 1),
                  "passes_p95": round(float(np.percentile(passes, 95)), 1), "passes_max": round(float(passes.max()), 1),
                  "top_source_p": round(float(t["source_p"].max()), 3),
                  "top_source_cap": round(float(t["caps"][int(np.argmax(t["source_p"]))]), 3),
                  "clips_at_pass_cap": int(np.sum(t["p"] >= t["clip_caps"] * (1 - 1e-6)))}
    return out

import argparse
import glob
import json
import os
import sys
import time
from collections import Counter

import numpy as np
import torch
import torchaudio.functional as AF

ROOT = os.path.abspath(os.environ.get("SARGE_ROOT", "D:/Python/SAGE"))
sys.path.insert(0, os.path.join(ROOT, "src"))

from sarge.generators.stable_audio import _extract
from sarge.loudness import normalize_loudness, target_for
from sarge.types import AudioClip

SR = 44100
DUR = 0.6
MATERIALS = ["wood", "stone", "grass", "metal"]
CATEGORIES = ["a footstep", "a gunshot", "music", "speech", "silence", "rain", "an explosion"]
DISTRACTORS = ["hands clapping", "typing on a keyboard", "a hammer hitting", "a ball bouncing",
               "a clock ticking", "a door closing"]
NEG_CLASSES = ["door_wood_knock", "glass_breaking", "water_drops", "crackling_fire", "wind"]
AST_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
CLAP_NAME = "laion/clap-htsat-unfused"
PANNS_LEN = 320000
CLAP_LEN = 480000
FITS = ["none", "silence", "tile", "tile_floor"]
FOOT = ("walk, footsteps",)
LOCO = ("walk, footsteps", "run", "shuffle")
REAL = "real_matched_0.6s"
GATE = [("clap_hard", "native"), ("ast", "tile"), ("panns", "tile")]
GATE_BASE = [("clap_base", "native"), ("ast", "tile"), ("panns", "tile")]


def rs(x, src, dst):
    x = np.ascontiguousarray(x, dtype=np.float32)
    if src == dst:
        return x
    return AF.resample(torch.from_numpy(x), src, dst).numpy().astype(np.float32)


def load(p):
    c = AudioClip.from_wav(p)
    return rs(c.samples, c.sample_rate, SR)


def level(x):
    return normalize_loudness(x, target_rms=target_for("normal"))


def onset(x, dur=DUR):
    env = np.abs(x)
    i = int(np.argmax(env > 0.1 * env.max())) if env.max() > 0 else 0
    return x[i: i + int(dur * SR)]


def matched(x):
    return level(_extract(x, SR, DUR))


def single(x, keep=0.30):
    y = _extract(x, SR, DUR).copy()
    a, b = int(keep * SR), int((keep + 0.05) * SR)
    y[a:b] *= np.linspace(1.0, 0.0, len(y[a:b]), dtype=np.float32)
    y[b:] = 0.0
    return level(y)


def click(seed):
    r = np.random.default_rng(seed)
    n, s = int(DUR * SR), int(0.03 * SR)
    x = np.zeros(n, np.float32)
    t = np.arange(n - s) / SR
    x[s:] = (r.standard_normal(n - s) * np.exp(-t / 0.02)).astype(np.float32)
    return level(x)


def fit(x, n, mode, seed=0):
    x = np.asarray(x, dtype=np.float32)
    if mode in ("none", "native") or len(x) == 0:
        return x
    if mode == "silence":
        y = np.pad(x[:n], (0, max(0, n - len(x))))
    else:
        y = np.tile(x, n // len(x) + 1)[:n]
    if mode == "tile_floor":
        y = y + np.random.default_rng(seed).standard_normal(n).astype(np.float32) * np.float32(1e-3)
    return y


def crest_db(x):
    rms = float(np.sqrt(np.mean(x ** 2))) + 1e-12
    return float(20 * np.log10(np.max(np.abs(x)) / rms + 1e-12))


def attack_ms(x):
    w = max(1, int(0.001 * SR))
    env = np.convolve(np.abs(x), np.ones(w) / w, mode="same")
    pk = int(np.argmax(env))
    p = env[pk]
    if p <= 0:
        return 0.0
    a = int(np.argmax(env[: pk + 1] >= 0.1 * p))
    b = int(np.argmax(env[: pk + 1] >= 0.9 * p))
    return (b - a) / SR * 1000.0


def n_steps(x, gap=0.15):
    w = max(1, int(0.005 * SR))
    env = np.convolve(np.abs(x), np.ones(w) / w, mode="same")
    if env.max() <= 0:
        return 0
    e = env[1:-1]
    idx = np.flatnonzero((e > 0.3 * env.max()) & (e >= env[:-2]) & (e >= env[2:])) + 1
    peaks = []
    for i in idx:
        if not peaks or i - peaks[-1] > gap * SR:
            peaks.append(int(i))
    return len(peaks)


def shape(clips):
    win = [onset(x, 0.3) for x in clips]
    win = [w for w in win if len(w) >= int(0.02 * SR) and np.abs(w).max() > 0]
    st = [n_steps(x) for x in clips]
    return {"crest_db": float(np.mean([crest_db(w) for w in win])) if win else float("nan"),
            "crest_sd": float(np.std([crest_db(w) for w in win])) if win else float("nan"),
            "attack_ms": float(np.median([attack_ms(w) for w in win])) if win else float("nan"),
            "steps_mean": float(np.mean(st)), "multi_step_frac": float(np.mean([s > 1 for s in st]))}


def build_sets(gen_dirs, n_ctl):
    real = [load(p) for p in sorted(glob.glob(os.path.join(ROOT, "data", "real_footsteps", "*.wav")))]
    sets = {"real_full_5s": real,
            "real_onset_0.6s": [onset(x) for x in real],
            REAL: [matched(x) for x in real],
            "real_single_0.6s": [single(x) for x in real]}
    for c in NEG_CLASSES:
        ps = sorted(glob.glob(os.path.join(ROOT, "data", "esc50", c, "*.wav")))
        if ps:
            sets["neg_" + c] = [matched(load(p)) for p in ps]
    labels = {}
    for d in gen_dirs:
        ps = sorted(glob.glob(os.path.join(d, "*_[0-9][0-9].wav")))
        if ps:
            name = "gen_" + os.path.basename(os.path.normpath(d))
            sets[name] = [load(p) for p in ps]
            labels[name] = [os.path.basename(p).split("_")[0] for p in ps]
    sets["ctl_click"] = [click(i) for i in range(n_ctl)]
    sets["ctl_silence"] = [np.zeros(int(DUR * SR), np.float32) for _ in range(2)]
    return sets, labels


def score_rows(logits, probs, lab, foot, loco):
    rows = []
    for lg, p in zip(logits, probs):
        top = [int(t) for t in np.argsort(lg)[::-1][:5]]
        rows.append({"hit": any(t in foot for t in top), "hit_loco": any(t in loco for t in top),
                     "p": float(p[foot].max()), "p_loco": float(p[loco].max()), "top1": lab[top[0]]})
    return rows


def run_ast(sets, dev, fits, bs):
    from transformers import ASTForAudioClassification, AutoFeatureExtractor
    m = ASTForAudioClassification.from_pretrained(AST_NAME).to(dev).eval()
    fe = AutoFeatureExtractor.from_pretrained(AST_NAME)
    lab = {int(i): n.lower() for i, n in m.config.id2label.items()}
    foot = [i for i, n in lab.items() if n in FOOT]
    loco = [i for i, n in lab.items() if n in LOCO]
    n = 400 + (fe.max_length - 1) * 160
    s16 = {k: [rs(x, SR, 16000) for x in v] for k, v in sets.items()}
    out = {}
    for f in fits:
        t0 = time.perf_counter()
        for name, clips in s16.items():
            xs = [fit(x, n, f, i) for i, x in enumerate(clips)]
            rows = []
            for i in range(0, len(xs), bs):
                inp = fe(xs[i: i + bs], sampling_rate=16000, return_tensors="pt")
                with torch.no_grad():
                    lg = m(**{k: v.to(dev) for k, v in inp.items()}).logits.float().cpu()
                rows += score_rows(lg.numpy(), torch.sigmoid(lg).numpy(), lab, foot, loco)
            out[("ast", f, name)] = rows
        print("ast %-10s %.1f s" % (f, time.perf_counter() - t0), flush=True)
    del m
    torch.cuda.empty_cache()
    return out


def run_panns(sets, dev, fits, bs):
    from panns_inference import AudioTagging
    from panns_inference.config import labels
    at = AudioTagging(checkpoint_path=None, device=dev)
    lab = {i: n.lower() for i, n in enumerate(labels)}
    foot = [i for i, n in lab.items() if n in FOOT]
    loco = [i for i, n in lab.items() if n in LOCO]
    s32 = {k: [rs(x, SR, 32000) for x in v] for k, v in sets.items()}
    out = {}
    for f in fits:
        t0 = time.perf_counter()
        for name, clips in s32.items():
            xs = [fit(x, PANNS_LEN, f, i) for i, x in enumerate(clips)]
            xs = [x if len(x) >= 16000 else np.pad(x, (0, 16000 - len(x))) for x in xs]
            rows = []
            i = 0
            while i < len(xs):
                j = i + 1
                while j < len(xs) and j - i < bs and len(xs[j]) == len(xs[i]):
                    j += 1
                cw, _ = at.inference(np.stack(xs[i:j]).astype(np.float32))
                rows += score_rows(cw, cw, lab, foot, loco)
                i = j
            out[("panns", f, name)] = rows
        print("panns %-10s %.1f s" % (f, time.perf_counter() - t0), flush=True)
    del at
    torch.cuda.empty_cache()
    return out


def softmax(z):
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def run_clap(sets, labels, dev, bs):
    from transformers import ClapModel, ClapProcessor
    m = ClapModel.from_pretrained(CLAP_NAME).to(dev).eval()
    pr = ClapProcessor.from_pretrained(CLAP_NAME)
    texts = CATEGORIES + DISTRACTORS
    mats = ["the sound of a footstep on %s" % x for x in MATERIALS]
    tt = pr(text=texts + mats, return_tensors="pt", padding=True)
    fi = CATEGORIES.index("a footstep")
    s48 = {k: [rs(x, SR, 48000) for x in v] for k, v in sets.items()}
    out = {}
    for f in ("native", "silence"):
        t0 = time.perf_counter()
        for name, clips in s48.items():
            xs = [fit(x, CLAP_LEN, f) for x in clips]
            lg = []
            for i in range(0, len(xs), bs):
                a = pr(audio=xs[i: i + bs], sampling_rate=48000, return_tensors="pt")
                with torch.no_grad():
                    lg.append(m(**{k: v.to(dev) for k, v in {**a, **tt}.items()}).logits_per_audio.float().cpu().numpy())
            lg = np.concatenate(lg)
            cat, mat = lg[:, :len(texts)], lg[:, len(texts):]
            labs = labels.get(name, [None] * len(xs))
            for tag, k in (("clap_base", len(CATEGORIES)), ("clap_hard", len(texts))):
                c = cat[:, :k]
                p = softmax(c)
                out[(tag, f, name)] = [
                    {"hit": bool(c[i].argmax() == fi), "p": float(p[i, fi]), "top1": texts[int(c[i].argmax())],
                     "mat": bool(mat[i].argmax() == MATERIALS.index(l)) if l in MATERIALS else None}
                    for i, l in enumerate(labs)]
        print("clap %-10s %.1f s" % (f, time.perf_counter() - t0), flush=True)
    del m
    torch.cuda.empty_cache()
    return out


def run_audiobox(sets, bs):
    from audiobox_aesthetics.infer import initialize_predictor
    p = initialize_predictor()
    out = {}
    t0 = time.perf_counter()
    for name, clips in sets.items():
        if name.startswith(("neg_", "ctl_")):
            continue
        rows = []
        for i in range(0, len(clips), bs):
            rows += p.forward([{"path": torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))[None],
                                "sample_rate": SR} for x in clips[i: i + bs]])
        out[("audiobox", "native", name)] = rows
    print("audiobox   %.1f s" % (time.perf_counter() - t0), flush=True)
    del p
    torch.cuda.empty_cache()
    return out


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def col(raw, key, name, field):
    return np.array([float(r[field]) for r in raw.get(key + (name,), [])], dtype=float)


def frac(v):
    return float(np.mean(v)) if len(v) else float("nan")


def analyze(raw, sets, fpr, min_real, min_gap):
    negs = [n for n in sets if n.startswith("neg_")]
    gens = [n for n in sets if n.startswith("gen_")]
    rep = {}
    for key in sorted({k[:2] for k in raw if k[0] != "audiobox"}):
        hit_fields = ["hit", "hit_loco"] if key[0] in ("ast", "panns") else ["hit"]
        for hf in hit_fields:
            pf = "p" if hf == "hit" else "p_loco"
            rate = {n: frac(col(raw, key, n, hf)) for n in sets}
            pneg = np.concatenate([col(raw, key, n, pf) for n in negs]) if negs else np.array([])
            hneg = np.concatenate([col(raw, key, n, hf) for n in negs]) if negs else np.array([])
            tau = float(np.quantile(pneg, 1.0 - fpr)) if len(pneg) else float("nan")
            above = {n: frac(col(raw, key, n, pf) > tau) for n in sets}
            rr, nr = rate[REAL], frac(hneg)
            r = {"rate": rate, "neg_rate": nr,
                 "neg_rate_by_class": {n: rate[n] for n in negs},
                 "auc_real_vs_neg": auc(col(raw, key, REAL, pf), pneg),
                 "tau": tau, "rate_above_tau": above,
                 "valid": bool(rr >= min_real and rr - nr >= min_gap),
                 "top1": {n: Counter(r_["top1"] for r_ in raw.get(key + (n,), [])).most_common(3) for n in sets},
                 "gen": {}}
            for g in gens:
                r["gen"][g] = {"rate": rate[g], "ratio": rate[g] / rr if rr > 0 else float("nan"),
                               "auc_vs_neg": auc(col(raw, key, g, pf), pneg),
                               "tau_ratio": above[g] / above[REAL] if above[REAL] > 0 else float("nan")}
            rep["%s|%s|%s" % (key[0], key[1], hf)] = r
    mat = {}
    for g in gens:
        v = [r_["mat"] for r_ in raw.get(("clap_base", "native", g), []) if r_["mat"] is not None]
        mat[g] = {"material_acc": frac(v), "n": len(v)}
    ab = {}
    for k, rows in raw.items():
        if k[0] == "audiobox":
            ab[k[2]] = {a: float(np.mean([r_[a] for r_ in rows])) for a in ("CE", "CU", "PC", "PQ")}
    return rep, mat, ab


def consensus(rep, gens, combo, real_key=None):
    out = {}
    for g in gens:
        vals, used = [], []
        for j, f in combo:
            r = rep.get("%s|%s|hit" % (j, f))
            if not r or not r["valid"]:
                continue
            rr = r["rate"][real_key] if real_key else r["rate"][REAL]
            if rr > 0:
                vals.append(min(r["rate"][g] / rr, 1.0))
                used.append(j)
        out[g] = {"consensus": min(vals) if vals else float("nan"), "judges": used}
    return out


def pct(v):
    return "  nan " if v != v else "%5.1f%%" % (100 * v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", nargs="+", default=None)
    ap.add_argument("--judges", nargs="+", default=["clap", "ast", "panns", "audiobox"])
    ap.add_argument("--fits", nargs="+", default=FITS)
    ap.add_argument("--fpr", type=float, default=0.1)
    ap.add_argument("--min-real", type=float, default=0.5)
    ap.add_argument("--min-gap", type=float, default=0.3)
    ap.add_argument("--ctl", type=int, default=10)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(ROOT, "scripts", "bench", "out_recalibrate.json"))
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gen_dirs = args.gen or sorted(d for d in glob.glob(os.path.join(ROOT, "data", "_gate", "*")) if os.path.isdir(d))
    sets, labels = build_sets(gen_dirs, args.ctl)
    print("sets: " + ", ".join("%s=%d" % (k, len(v)) for k, v in sets.items()), flush=True)

    raw = {}
    if "ast" in args.judges:
        raw.update(run_ast(sets, dev, args.fits, args.bs))
    if "panns" in args.judges:
        raw.update(run_panns(sets, dev, args.fits, 2 * args.bs))
    if "clap" in args.judges:
        raw.update(run_clap(sets, labels, dev, 4 * args.bs))
    if "audiobox" in args.judges:
        raw.update(run_audiobox(sets, 2 * args.bs))

    rep, mat, ab = analyze(raw, sets, args.fpr, args.min_real, args.min_gap)
    gens = [n for n in sets if n.startswith("gen_")]
    shapes = {n: shape(v) for n, v in sets.items() if not n.startswith("ctl_silence")}
    cons = {"hard": consensus(rep, gens, GATE), "base": consensus(rep, gens, GATE_BASE),
            "hard_single": consensus(rep, gens, GATE, "real_single_0.6s")}

    cols = ["real_full_5s", "real_onset_0.6s", REAL, "real_single_0.6s"]
    print()
    print("=" * 112)
    print(" JUDGE VALIDITY ON REAL AUDIO   (hit = top-5 AudioSet label or CLAP argmax; neg = %d ESC-50 negatives)"
          % sum(len(v) for k, v in sets.items() if k.startswith("neg_")))
    print("=" * 112)
    print(" %-26s %8s %8s %8s %8s %8s %6s %6s  %s" % ("judge|fit|rule", "full5s", "onset", "matched", "single",
                                                     "neg", "AUC", "valid", "top1 on matched real"))
    for k, r in rep.items():
        print(" %-26s %8s %8s %8s %8s %8s %6.2f %6s  %s" % (
            k, pct(r["rate"]["real_full_5s"]), pct(r["rate"]["real_onset_0.6s"]), pct(r["rate"][REAL]),
            pct(r["rate"]["real_single_0.6s"]), pct(r["neg_rate"]), r["auc_real_vs_neg"],
            "yes" if r["valid"] else "no", [t for t, _ in r["top1"][REAL]]))
    print("-" * 112)
    for k in ("ast|none|hit", "panns|none|hit", "ast|tile|hit"):
        if k in rep:
            print(" %-16s top1 on ctl_silence %s   on ctl_click %s" % (
                k, rep[k]["top1"].get("ctl_silence"), rep[k]["top1"].get("ctl_click")))
    print("=" * 112)
    print(" GENERATED SETS   rate (gen/real ratio) per judge; negatives-calibrated tau ratio in brackets")
    print("=" * 112)
    for g in gens:
        print(" %s  n=%d  material %s  %s" % (g, len(sets[g]), pct(mat[g]["material_acc"]),
                                              "  ".join("%s %.2f" % (a, v) for a, v in ab.get(g, {}).items())))
        for k, r in rep.items():
            if k.endswith("|hit"):
                x = r["gen"][g]
                print("     %-26s %s (%.2f) [%.2f]  AUC vs neg %.2f%s" % (
                    k, pct(x["rate"]), x["ratio"], x["tau_ratio"], x["auc_vs_neg"], "" if r["valid"] else "   (judge invalid)"))
        print("     consensus hard %.2f %s | base %.2f %s | hard vs single-step anchor %.2f" % (
            cons["hard"][g]["consensus"], cons["hard"][g]["judges"], cons["base"][g]["consensus"],
            cons["base"][g]["judges"], cons["hard_single"][g]["consensus"]))
    print("=" * 112)
    print(" SHAPE   crest / attack / multi-step fraction")
    for n, s in shapes.items():
        if not n.startswith("neg_"):
            print("     %-22s crest %5.1f +- %4.1f dB  attack %5.2f ms  steps %.2f  multi %s" % (
                n, s["crest_db"], s["crest_sd"], s["attack_ms"], s["steps_mean"], pct(s["multi_step_frac"])))
    for n in ("real_full_5s", REAL, "real_single_0.6s"):
        if n in ab:
            print("     audiobox %-18s %s" % (n, "  ".join("%s %.2f" % (a, v) for a, v in ab[n].items())))
    print("=" * 112)

    per_clip = {"|".join(k): [r_.get("p", r_.get("PQ")) for r_ in v] for k, v in raw.items()}
    json.dump({"meta": {"root": ROOT, "sets": {k: len(v) for k, v in sets.items()}, "fpr": args.fpr,
                        "min_real": args.min_real, "min_gap": args.min_gap, "fits": args.fits,
                        "distractors": DISTRACTORS, "neg_classes": NEG_CLASSES},
               "judges": rep, "material": mat, "audiobox": ab, "shape": shapes, "consensus": cons,
               "per_clip": per_clip},
              open(args.out, "w"), indent=1)
    print("saved -> %s" % args.out)


if __name__ == "__main__":
    main()

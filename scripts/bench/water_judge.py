import os
import sys

BENCH = os.path.join(os.getcwd(), "scripts", "bench")
if not os.path.isfile(os.path.join(BENCH, "_bootstrap.py")):
    BENCH = r"D:\Python\SAGE\scripts\bench"
sys.path.insert(0, BENCH)

import _bootstrap

import argparse
import gc
import glob
import json
import re
import time
from collections import Counter

import numpy as np

from sarge.generators.stable_audio import _extract
from sarge.loudness import normalize_loudness, target_for
from sarge.types import AudioClip
from validate_panel import AST_NAME, PANNS_SR, _resample
from recalibrate import CLAP_LEN, CLAP_NAME, PANNS_LEN, auc, fit, softmax
from ab_clap import mcnemar

try:
    from water_probe import STATIC_FLAT_LT, STATIC_RISE_DB, metrics as probe_metrics
    PROBE_IMPORT_ERROR = None
except Exception as exc:
    probe_metrics, STATIC_FLAT_LT, STATIC_RISE_DB = None, None, None
    PROBE_IMPORT_ERROR = repr(exc)

ROOT = os.path.dirname(os.path.dirname(BENCH))
SR = 44100
REAL_CLASS = "water_drops"
NEG_CLASSES = ["crackling_fire", "wind", "door_wood_knock", "glass_breaking", "footsteps"]
NOISE_CLASSES = ["crackling_fire", "wind"]
SHORT = {"crackling_fire": "fire", "wind": "wind", "door_wood_knock": "knock", "glass_breaking": "glass",
         "footsteps": "foot", "static": "hiss"}
CTL_LENGTHS = [0.6, 1.0, 1.5]
TAG_FITS = ["none", "silence", "tile", "tile_floor"]
CLAP_FITS = ["native", "silence"]
CLAP_DEFAULT_FITS = ["native"]
RULES = ("top", "thr")
TAG_SETS = {
    "water_all": ["Water", "Rain", "Raindrop", "Rain on surface", "Stream", "Waterfall", "Ocean", "Waves, surf",
                  "Gurgling", "Liquid", "Splash, splatter", "Slosh", "Drip", "Pour", "Trickle, dribble", "Gush",
                  "Fill (with liquid)", "Boiling", "Plop", "Water tap, faucet", "Sink (filling or washing)",
                  "Bathtub (filling or washing)", "Toilet flush"],
    "liquid": ["Water", "Raindrop", "Gurgling", "Liquid", "Splash, splatter", "Slosh", "Drip", "Pour",
               "Trickle, dribble", "Gush", "Fill (with liquid)", "Boiling", "Plop"],
    "splash": ["Splash, splatter", "Slosh", "Plop"],
    "drip": ["Drip", "Raindrop"],
}
CLAP_WATER = ["water splashing", "something dropping into water", "a splash", "water", "liquid sloshing"]
CLAP_DIAG = ["water dripping"]
CLAP_DISTRACTORS = ["white noise static", "wind", "fire crackling", "a footstep", "a knock", "an explosion",
                    "glass breaking", "music", "speech"]
DIAGNOSTIC = {"drip", "water_dripping"}
HISS = "static"
HISS_SPECTRA = [(0.0, 50.0), (0.5, 50.0), (1.0, 200.0), (1.25, 400.0)]
HISS_PER = 10
REF = "L0_1.0_0.6|current"
PROBE_JSON = os.path.join("scripts", "bench", "out_water_probe.json")
PROBE_DIR = os.path.join("data", "_gate", "water_depth")
TIER_DIR = os.path.join("data", "_gate", "tier_water")
CLIP_NAME = re.compile(r"^([a-z]+)-([a-z]+)_(\d+)\.wav$")
NAN = float("nan")


def wk(length):
    return "%.1f" % float(length)


def cname(cls, w):
    return "ctl:%s@%s" % (cls, w)


def slug(text):
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def frac(v):
    v = np.asarray(v, dtype=float)
    return float(v.mean()) if v.size else NAN


def pct(v):
    return "  nan" if v != v else "%4.0f%%" % (100 * v)


def load(p):
    c = AudioClip.from_wav(p)
    return _resample(c.samples, c.sample_rate, SR)


def level(x):
    return normalize_loudness(x, target_rms=target_for("normal"))


def onset_crop(x, dur):
    return level(_extract(x, SR, dur))


def parse_clip(p):
    m = CLIP_NAME.match(os.path.basename(p))
    return ("%s|%s" % (m.group(1), m.group(2)), int(m.group(3))) if m else None


def clap_texts():
    return CLAP_WATER + CLAP_DIAG + CLAP_DISTRACTORS


def hiss(seed, alpha, hp, dur=5.0):
    n = int(dur * SR)
    s = np.fft.rfft(np.random.default_rng(seed).standard_normal(n))
    f = np.fft.rfftfreq(n, 1.0 / SR)
    g = np.zeros_like(f)
    g[1:] = (f[1:] >= hp) * np.maximum(f[1:], hp) ** (-alpha / 2.0)
    return np.fft.irfft(s * g, n).astype(np.float32)


def build_hiss(args, sets, meta):
    spec = [(a, hp) for a, hp in HISS_SPECTRA for _ in range(HISS_PER)]
    raw = [hiss(1000 + j, a, hp) for j, (a, hp) in enumerate(spec)]
    for length in args.ctl_lengths:
        w = wk(length)
        sets[cname(HISS, w)] = [onset_crop(x, length) for x in raw]
        meta[cname(HISS, w)] = {"kind": "hiss", "cls": HISS, "win": w, "spectra": spec}
    sets[cname(HISS, "full")] = [level(x) for x in raw]
    meta[cname(HISS, "full")] = {"kind": "hiss", "cls": HISS, "win": "full", "spectra": spec}


def build_controls(args):
    sets, meta = {}, {}
    for cls in [args.real_class] + list(args.neg_classes):
        ps = sorted(glob.glob(os.path.join(ROOT, "data", "esc50", cls, "*.wav")))
        if not ps:
            raise SystemExit("no clips under data/esc50/%s" % cls)
        raw = [load(p) for p in ps]
        kind = "real" if cls == args.real_class else "neg"
        for length in args.ctl_lengths:
            w = wk(length)
            sets[cname(cls, w)] = [onset_crop(x, length) for x in raw]
            meta[cname(cls, w)] = {"kind": kind, "cls": cls, "win": w, "paths": ps}
        sets[cname(cls, "full")] = raw
        meta[cname(cls, "full")] = {"kind": kind, "cls": cls, "win": "full", "paths": ps}
    build_hiss(args, sets, meta)
    return sets, meta


def assign_window(dur, lengths, tol=0.05):
    best = min(lengths, key=lambda length: abs(length - dur))
    return wk(best) if abs(best - dur) <= tol else None


def add_gen(sets, meta, name, d, args, extra):
    ps = sorted(p for p in glob.glob(os.path.join(d, "*.wav")) if parse_clip(p))
    if not ps:
        return
    clips = [load(p) for p in ps]
    dur = float(np.median([len(x) for x in clips])) / SR
    sets[name] = clips
    meta[name] = {"kind": "gen", "win": assign_window(dur, args.ctl_lengths), "dur": dur,
                  "paths": [os.path.relpath(p, ROOT) for p in ps], "pairs": [parse_clip(p) for p in ps], **extra}


def build_generated(args):
    sets, meta = {}, {}
    pj = os.path.join(ROOT, args.probe_json)
    probe = json.load(open(pj)) if os.path.isfile(pj) else None
    if probe:
        cfgs = list(probe["summary"])
        rows = {(r["cfg"], r["key"], int(r["v"])): r for r in probe["rows"]}
    else:
        cfgs = []
        for d in sorted(glob.glob(os.path.join(ROOT, args.probe_dir, "*"))):
            b = os.path.basename(d)
            if os.path.isdir(d) and not b.endswith("_crop06"):
                p = b.split("_", 3)
                cfgs.append("%s|%s" % ("_".join(p[:3]), p[3]))
        rows = {}
    for cfg in cfgs:
        lname, wname = cfg.split("|")
        base = os.path.join(ROOT, args.probe_dir, "%s_%s" % (lname, wname))
        for suffix, d, field in (("", base, "static"), ("@crop06", base + "_crop06", "c_static")):
            name = "gen:%s%s" % (cfg, suffix)
            add_gen(sets, meta, name, d, args, {"group": "probe", "cfg": cfg, "len": lname, "word": wname,
                                                "suffix": suffix, "len_ref": "gen:%s|current%s" % (lname, suffix)})
            if name in meta:
                meta[name]["json_static"] = [rows[(cfg, k, v)].get(field) if (cfg, k, v) in rows else None
                                             for k, v in meta[name]["pairs"]]
    tiers = sorted(d for d in glob.glob(os.path.join(ROOT, args.tier_dir, "*")) if os.path.isdir(d))
    for d in tiers:
        b = os.path.basename(d)
        add_gen(sets, meta, "gen:tier|%s" % b, d, args,
                {"group": "tier", "cfg": "tier|%s" % b, "len": "tier", "word": b, "suffix": "",
                 "len_ref": "gen:tier|%s" % os.path.basename(tiers[0])})
    return sets, meta, probe


def static_rates(sets, meta):
    out = {}
    for name, clips in sets.items():
        r = {"n": len(clips), "recomputed": None, "json": None, "agree": None}
        if probe_metrics is not None:
            s = [int(probe_metrics(x, SR)["static"]) for x in clips]
            r["recomputed"], r["per_clip"] = frac(s), s
        js = meta[name].get("json_static")
        if js and all(v is not None for v in js):
            r["json"] = frac(js)
            if "per_clip" in r:
                r["agree"] = int(sum(int(a) == int(b) for a, b in zip(r["per_clip"], js)))
        out[name] = r
    return out


def static_of(st):
    return st["recomputed"] if st["recomputed"] is not None else st["json"]


def _torch(threads):
    import torch
    torch.set_num_threads(threads)
    return torch, ("cuda" if torch.cuda.is_available() else "cpu")


def _release(torch):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _runs(xs, bs):
    i = 0
    while i < len(xs):
        j = i + 1
        while j < len(xs) and j - i < bs and len(xs[j]) == len(xs[i]):
            j += 1
        yield i, j
        i = j


def run_ast(sets, args):
    torch, dev = _torch(args.threads)
    from transformers import ASTForAudioClassification, AutoFeatureExtractor
    m = ASTForAudioClassification.from_pretrained(AST_NAME).to(dev).eval()
    fe = AutoFeatureExtractor.from_pretrained(AST_NAME)
    lab = {int(i): str(n) for i, n in m.config.id2label.items()}
    names = [lab[i] for i in range(len(lab))]
    n = 400 + (fe.max_length - 1) * 160
    s16 = {k: [_resample(x, SR, 16000) for x in v] for k, v in sets.items()}
    out = {}
    for f in args.tag_fits:
        t0 = time.perf_counter()
        out[f] = {}
        for name, clips in s16.items():
            probs = []
            for i in range(0, len(clips), args.bs):
                xs = [fit(x, n, f, i + k) for k, x in enumerate(clips[i: i + args.bs])]
                inp = fe(xs, sampling_rate=16000, return_tensors="pt")
                with torch.no_grad():
                    lg = m(**{k: v.to(dev) for k, v in inp.items()}).logits.float().cpu()
                probs.append(torch.sigmoid(lg).numpy().astype(np.float32))
            out[f][name] = np.concatenate(probs)
        print("ast   fit %-10s %7.1f s" % (f, time.perf_counter() - t0), flush=True)
    del m, fe
    _release(torch)
    return {"labels": names, "scores": out}


def run_panns(sets, args):
    torch, dev = _torch(args.threads)
    from panns_inference import AudioTagging
    from panns_inference.config import labels
    at = AudioTagging(checkpoint_path=None, device=dev)
    s32 = {k: [_resample(x, SR, PANNS_SR) for x in v] for k, v in sets.items()}
    bs = 2 * args.bs
    out = {}
    for f in args.tag_fits:
        t0 = time.perf_counter()
        out[f] = {}
        for name, clips in s32.items():
            probs = []
            for i in range(0, len(clips), bs):
                xs = [fit(x, PANNS_LEN, f, i + k) for k, x in enumerate(clips[i: i + bs])]
                xs = [x if len(x) >= 16000 else np.pad(x, (0, 16000 - len(x))) for x in xs]
                for a, b in _runs(xs, bs):
                    cw, _ = at.inference(np.stack(xs[a:b]).astype(np.float32))
                    probs.append(np.asarray(cw, dtype=np.float32))
            out[f][name] = np.concatenate(probs)
        print("panns fit %-10s %7.1f s" % (f, time.perf_counter() - t0), flush=True)
    del at
    _release(torch)
    return {"labels": [str(x) for x in labels], "scores": out}


def run_clap(sets, args):
    torch, dev = _torch(args.threads)
    from transformers import ClapModel, ClapProcessor
    m = ClapModel.from_pretrained(CLAP_NAME).to(dev).eval()
    pr = ClapProcessor.from_pretrained(CLAP_NAME)
    texts = clap_texts()
    tt = pr(text=texts, return_tensors="pt", padding=True)
    s48 = {k: [_resample(x, SR, 48000) for x in v] for k, v in sets.items()}
    bs = 4 * args.bs
    out = {}
    for f in args.clap_fits:
        t0 = time.perf_counter()
        out[f] = {}
        for name, clips in s48.items():
            lg = []
            for i in range(0, len(clips), bs):
                xs = [fit(x, CLAP_LEN, f) for x in clips[i: i + bs]]
                a = pr(audio=xs, sampling_rate=48000, return_tensors="pt")
                with torch.no_grad():
                    lg.append(m(**{k: v.to(dev) for k, v in {**a, **tt}.items()}).logits_per_audio.float().cpu().numpy())
            out[f][name] = np.concatenate(lg).astype(np.float32)
            print("  clap %-8s %-40s %3d clips %7.1f s" % (f, name, len(clips), time.perf_counter() - t0), flush=True)
        print("clap  fit %-10s %7.1f s" % (f, time.perf_counter() - t0), flush=True)
    del m, pr
    _release(torch)
    return {"labels": texts, "scores": out}


JUDGES = {"ast": run_ast, "panns": run_panns, "clap": run_clap}


def resolve(labels, label_sets):
    low = {str(n).lower(): i for i, n in enumerate(labels)}
    out = {}
    for s, names in label_sets.items():
        miss = [n for n in names if n.lower() not in low]
        if miss:
            raise SystemExit("label set %s names missing from the model label list: %s" % (s, miss))
        out[s] = [low[n.lower()] for n in names]
    return out


def tagger_variants(fam, res):
    labels = res["labels"]
    ids = resolve(labels, TAG_SETS)
    out = {}
    for f, by in res["scores"].items():
        tops = {name: np.argsort(-P, axis=1)[:, :5] for name, P in by.items()}
        for s, ix in ids.items():
            ixa = np.asarray(ix)
            d = {}
            for name, P in by.items():
                sub = P[:, ixa]
                d[name] = {"hit": np.isin(tops[name], ixa).any(1), "score": sub.sum(1).astype(float),
                           "lead": [labels[ix[j]] for j in sub.argmax(1)],
                           "top1": [labels[j] for j in tops[name][:, 0]]}
            out["%s|%s|%s" % (fam, f, s)] = d
    return out, {s: [{"index": i, "name": labels[i]} for i in ix] for s, ix in ids.items()}


def clap_variants(res):
    texts = list(res["labels"])
    dis = [texts.index(t) for t in CLAP_DISTRACTORS]
    groups = {slug(p): [texts.index(p)] for p in CLAP_WATER + CLAP_DIAG}
    groups["any"] = [texts.index(p) for p in CLAP_WATER]
    out = {}
    for f, by in res["scores"].items():
        for g, w in groups.items():
            cols = w + dis
            d = {}
            for name, L in by.items():
                p = softmax(L[:, cols])
                am = p.argmax(1)
                d[name] = {"hit": am < len(w), "score": p[:, :len(w)].sum(1).astype(float),
                           "lead": [texts[w[int(j)]] for j in p[:, :len(w)].argmax(1)],
                           "top1": [texts[cols[int(j)]] for j in am]}
            out["clap|%s|%s" % (f, g)] = d
    return out, {g: [texts[i] for i in w] for g, w in groups.items()}


def hits(d, rule, tau):
    return np.asarray(d["hit"], dtype=bool) if rule == "top" else np.asarray(d["score"]) > tau


def evaluate(var, args):
    wins = [wk(x) for x in args.ctl_lengths] + ["full"]
    ctl, taus = {}, {}
    for base, by in var.items():
        for w in wins:
            real = by.get(cname(args.real_class, w))
            negs = {c: by.get(cname(c, w)) for c in args.neg_classes}
            hz = by.get(cname(HISS, w))
            if real is None or hz is None or any(v is None for v in negs.values()):
                continue
            pool = np.concatenate([n["score"] for n in negs.values()])
            noise = [negs[c]["score"] for c in args.noise_classes if c in negs]
            tau = float(np.quantile(pool, 1.0 - args.fpr))
            taus[(base, w)] = tau
            a = {"pooled": auc(real["score"], pool),
                 "noise": auc(real["score"], np.concatenate(noise)) if noise else NAN,
                 "hiss": auc(real["score"], hz["score"]),
                 **{c: auc(real["score"], n["score"]) for c, n in negs.items()}}
            for rule in RULES:
                cr = {c: frac(hits(n, rule, tau)) for c, n in negs.items()}
                nr = frac(np.concatenate([hits(n, rule, tau) for n in negs.values()]))
                rr = frac(hits(real, rule, tau))
                hr = frac(hits(hz, rule, tau))
                valid = bool(rr >= args.min_real and rr - nr >= args.min_gap)
                hiss_ok = bool(rr - hr >= args.min_gap)
                ctl.setdefault("%s|%s" % (base, rule), {})[w] = {
                    "real": rr, "neg": nr, "neg_by_class": cr, "hiss": hr, "auc": a, "tau": tau,
                    "real_lead": Counter(real["lead"]).most_common(3),
                    "real_top1": Counter(real["top1"]).most_common(3),
                    "hiss_top1": Counter(hz["top1"]).most_common(3),
                    "valid": valid,
                    "strict": bool(rr >= args.min_real and rr - max(cr.values()) >= args.min_gap),
                    "hiss_ok": hiss_ok, "usable": valid and hiss_ok}
    return ctl, taus


def paired(a, b):
    A, B = dict(a), dict(b)
    ks = [k for k in B if k in A]
    x = np.array([A[k] for k in ks], dtype=bool)
    y = np.array([B[k] for k in ks], dtype=bool)
    g, l = int((x & ~y).sum()), int((~x & y).sum())
    return {"n": len(ks), "rate": frac(x), "ref_rate": frac(y), "gain": g, "loss": l, "p": mcnemar(g, l)}


def score_generated(var, ctl, taus, meta, args):
    gens = [n for n, m in meta.items() if m["kind"] == "gen"]
    out = {}
    for v, by_w in ctl.items():
        base, rule = v.rsplit("|", 1)
        for g in gens:
            w = meta[g]["win"]
            c = by_w.get(w) if w else None
            if not c or not c["usable"]:
                continue
            d = var[base][g]
            h = hits(d, rule, taus[(base, w)])
            pool = np.concatenate([var[base][cname(x, w)]["score"] for x in args.neg_classes])
            noise = [var[base][cname(x, w)]["score"] for x in args.noise_classes if x in args.neg_classes]
            keys = [p[0] for p in meta[g]["pairs"]]
            rate = frac(h)
            out.setdefault(v, {})[g] = {
                "win": w, "n": int(len(h)), "rate": rate, "real": c["real"],
                "ratio": rate / c["real"] if c["real"] > 0 else NAN,
                "auc_vs_neg": auc(d["score"], pool),
                "auc_vs_noise": auc(d["score"], np.concatenate(noise)) if noise else NAN,
                "by_key": {k: frac(h[np.array([x == k for x in keys])]) for k in sorted(set(keys))},
                "lead": Counter(l for l, x in zip(d["lead"], h) if x).most_common(3),
                "top1": Counter(d["top1"]).most_common(3),
                "pairs": list(zip(meta[g]["pairs"], h.tolist()))}
    ref = "gen:" + args.ref
    for v, by_g in out.items():
        for g, rec in by_g.items():
            if ref in by_g and g != ref:
                rec["vs_ref"] = {**paired(rec["pairs"], by_g[ref]["pairs"]), "ref": ref,
                                 "cross_window": meta[g]["win"] != meta[ref]["win"]}
            lr = meta[g].get("len_ref")
            if lr and lr in by_g and lr not in (g, ref):
                rec["vs_len"] = {**paired(rec["pairs"], by_g[lr]["pairs"]), "ref": lr}
    return out


def rank(gen, ctl, meta, static):
    wins = sorted({m["win"] for m in meta.values() if m["kind"] == "gen" and m["win"]}, key=float)
    unjudged = sorted(g for g, m in meta.items() if m["kind"] == "gen" and not m["win"])
    chosen, no_valid, ranking, usable_list = {}, {}, {}, {}
    for w in wins:
        sets_w = [g for g, m in meta.items() if m["kind"] == "gen" and m["win"] == w]
        usable_list[w] = [v for v, by in ctl.items() if w in by and by[w]["usable"]]
        cand = [v for v in usable_list[w] if v.split("|")[2] not in DIAGNOSTIC]
        if not cand:
            no_valid[w] = sets_w
            continue
        pick = {}
        for v in cand:
            c = ctl[v][w]
            key = (c["strict"], c["auc"]["pooled"], c["real"] - c["neg"])
            fam = v.split("|")[0]
            if fam not in pick or key > pick[fam][0]:
                pick[fam] = (key, v)
        chosen[w] = {f: v for f, (_k, v) in sorted(pick.items())}
        rows = []
        for g in sets_w:
            per = {f: gen[v][g] for f, v in chosen[w].items()}
            ratios = [r["ratio"] for r in per.values()]
            st = static_of(static[g])
            rows.append({"set": g, "consensus": min(min(r, 1.0) for r in ratios), "mean_ratio": frac(ratios),
                         "static": st, "rates": {f: r["rate"] for f, r in per.items()}})
        rows.sort(key=lambda r: (-r["consensus"], -r["mean_ratio"], r["static"] if r["static"] is not None else 1.0))
        ranking[w] = rows
    return {"windows": wins, "chosen": chosen, "no_valid": no_valid, "unjudged": unjudged,
            "ranking": ranking, "usable": usable_list}


def label(g):
    return g[4:] if g.startswith("gen:") else g


def mc(x):
    if not x:
        return "-"
    return "%+d/-%d p%.2f%s" % (x["gain"], x["loss"], x["p"], "x" if x.get("cross_window") else "")


def report(sets, meta, var, ctl, gen, rk, static, labelsets, args):
    W = 150
    wins = [wk(x) for x in args.ctl_lengths] + ["full"]
    negs = list(args.neg_classes)
    nn = sum(len(sets[cname(c, wins[0])]) for c in negs)
    print()
    print("=" * W)
    print(" WATER JUDGE PANEL   positive control = ESC-50 %s (%d clips), negatives = %s (%d clips pooled)"
          % (args.real_class, len(sets[cname(args.real_class, wins[0])]), ", ".join(negs), nn))
    print(" control crops: _extract onset window then normalize_loudness at target_for('normal'), plus the full 5 s files")
    print(" top = any water label in the AudioSet top 5 (AST, PANNs) or a water phrase as CLAP argmax;"
          " thr = summed water probability above the %.0fth percentile of the pooled negatives at that length"
          % (100 * (1 - args.fpr)))
    print(" valid = real >= %.2f and real - pooled neg >= %.2f;  Y = also real - every single negative class >= %.2f,"
          " y = pooled rule only, - = invalid" % (args.min_real, args.min_gap, args.min_gap))
    print(" hiss = %d synthetic static clips (spectral slope, high-pass Hz: %s) cut the same way; h = valid but real - hiss"
          " < %.2f, so the judge calls static water and never scores generated clips"
          % (len(sets[cname(HISS, wins[0])]), ", ".join("%.2f/%d" % s for s in HISS_SPECTRA), args.min_gap))
    print("=" * W)
    for fam, ls in labelsets.items():
        for s, items in ls.items():
            names = [x["name"] if isinstance(x, dict) else x for x in items]
            print(" %-6s %-15s %s%s" % (fam, s, "; ".join(names), "   (diagnostic, never ranked)" if s in DIAGNOSTIC else ""))
    bases = sorted({v.rsplit("|", 1)[0] for v in ctl})
    bw = max([34] + [len(b) for b in bases])
    for w in wins:
        print("-" * W)
        print(" VALIDITY at %s%s" % (w, "" if w == "full" else " s"))
        hdr = " %-" + str(bw) + "s | top: %5s %5s " + " ".join(["%5s"] * len(negs)) + " %5s %2s | thr: %5s " + \
              " ".join(["%5s"] * len(args.noise_classes)) + " %5s %2s | %5s %5s %5s"
        print(hdr % tuple(["judge|fit|set", "real", "neg"] + [SHORT.get(c, c[:5]) for c in negs] + ["hiss", "ok", "real"] +
                          [SHORT.get(c, c[:5]) for c in args.noise_classes] + ["hiss", "ok", "AUC", "AUCnz", "AUChs"]))
        for b in bases:
            t, h = ctl.get(b + "|top", {}).get(w), ctl.get(b + "|thr", {}).get(w)
            if not t or not h:
                continue
            ok = lambda c: "-" if not c["valid"] else ("h" if not c["hiss_ok"] else ("Y" if c["strict"] else "y"))
            print(hdr % tuple([b, pct(t["real"]), pct(t["neg"])] + [pct(t["neg_by_class"][c]) for c in negs] +
                              [pct(t["hiss"]), ok(t), pct(h["real"])] + [pct(h["neg_by_class"][c]) for c in args.noise_classes] +
                              [pct(h["hiss"]), ok(h), "%.2f" % t["auc"]["pooled"], "%.2f" % t["auc"]["noise"],
                               "%.2f" % t["auc"]["hiss"]]))
        tops = [b for b in bases if not b.startswith("clap") and b.endswith("|water_all")]
        for b in tops:
            c = ctl[b + "|top"].get(w)
            if c:
                print("   %-30s top1 on real water: %s   on hiss: %s" % (
                    b.rsplit("|", 1)[0], ", ".join("%s %d" % x for x in c["real_top1"]),
                    ", ".join("%s %d" % x for x in c["hiss_top1"])))
    print("=" * W)
    print(" STATIC (water_probe rule: long-term flatness >= %s and onset rise <= %s dB)%s" % (
        STATIC_FLAT_LT, STATIC_RISE_DB, "" if probe_metrics else "   recompute unavailable: %s" % PROBE_IMPORT_ERROR))
    for w in wins:
        row = []
        for c in [args.real_class] + negs + [HISS]:
            st = static[cname(c, w)]
            row.append("%s %s" % (SHORT.get(c, c[:5]), pct(st["recomputed"]) if st["recomputed"] is not None else "n/a"))
        print("   controls @%-5s %s" % (w, "   ".join(row)))
    for g in [n for n, m in meta.items() if m["kind"] == "gen"]:
        st = static[g]
        print("   %-34s win %-5s recomputed %s   probe json %s   agree %s" % (
            label(g), meta[g]["win"] or "none", pct(st["recomputed"]) if st["recomputed"] is not None else "  n/a",
            pct(st["json"]) if st["json"] is not None else "  n/a",
            "%d/%d" % (st["agree"], st["n"]) if st["agree"] is not None else "-"))
    for w in rk["windows"]:
        print("=" * W)
        sets_w = [g for g, m in meta.items() if m["kind"] == "gen" and m["win"] == w]
        print(" GENERATED at the %s s window: %d sets, usable judge variants %d (valid and passing the hiss check)" % (
            w, len(sets_w), len(rk["usable"][w])))
        if w in rk["no_valid"]:
            print(" NO VALID JUDGE at %s s: every non-diagnostic judge fails the real water control or calls synthetic"
                  " static water at this length, so nothing is ranked here (%s)" % (
                      w, ", ".join(label(g) for g in rk["no_valid"][w])))
            continue
        ch = rk["chosen"][w]
        for f, v in ch.items():
            c = ctl[v][w]
            print("   %-6s %-48s real %s  neg %s  fire %s  wind %s  hiss %s  AUC %.2f  AUCnz %.2f  AUChs %.2f  %s" % (
                f, v, pct(c["real"]), pct(c["neg"]), pct(c["neg_by_class"].get("crackling_fire", NAN)),
                pct(c["neg_by_class"].get("wind", NAN)), pct(c["hiss"]), c["auc"]["pooled"], c["auc"]["noise"],
                c["auc"]["hiss"], "strict" if c["strict"] else "pooled only"))
        print("   other usable variants: %s" % (", ".join(v for v in rk["usable"][w] if v not in ch.values()) or "none"))
        hs = sorted(v for v, by in ctl.items() if w in by and by[w]["valid"] and not by[w]["hiss_ok"])
        print("   valid on real water but failing the hiss check: %s" % (", ".join(hs) or "none"))
        print(" RANKING (cons = min over chosen judges of min(rate / real rate, 1); McNemar vs %s paired on key+seed,"
              " x = different window)" % args.ref)
        fams = list(ch)
        hdr = " %4s %-32s %6s %6s %7s" + " | %-30s" * len(fams)
        print(hdr % tuple(["rank", "set", "cons", "mean", "static"] + ["%s rate(ratio) vs ref" % f for f in fams]))
        for i, r in enumerate(rk["ranking"][w]):
            cells = []
            for f in fams:
                x = gen[ch[f]][r["set"]]
                cells.append("%s(%.2f) %s" % (pct(x["rate"]), x["ratio"], mc(x.get("vs_ref"))))
            print(hdr % tuple([i + 1, label(r["set"]), "%.2f" % r["consensus"], "%.2f" % r["mean_ratio"],
                               pct(r["static"]) if r["static"] is not None else "  n/a"] + cells))
        keys = sorted({k for g in sets_w for k, _v in meta[g]["pairs"]})
        print(" PER KEY rate per chosen judge (the control is drops, so compare splash keys against footstep keys)")
        print(" %-32s %-7s " % ("set", "judge") + " ".join("%15s" % k for k in keys) + "   lead water label on hits")
        for r in rk["ranking"][w]:
            for f in fams:
                x = gen[ch[f]][r["set"]]
                print(" %-32s %-7s " % (label(r["set"]), f) + " ".join("%15s" % pct(x["by_key"].get(k, NAN)) for k in keys)
                      + "   " + ", ".join("%s %d" % y for y in x["lead"]))
        print(" WITHIN LENGTH (vs the same length with current wording, same window)")
        for r in rk["ranking"][w]:
            cells = [mc(gen[ch[f]][r["set"]].get("vs_len")) for f in fams]
            if any(c != "-" for c in cells):
                print("   %-32s %s" % (label(r["set"]), "   ".join("%s %s" % (f, c) for f, c in zip(fams, cells))))
    if rk["unjudged"]:
        print(" UNJUDGED (no control at this clip length): %s" % ", ".join(label(g) for g in rk["unjudged"]))
    gw = set(rk["windows"])
    print("=" * W)
    print(" VERDICT: " + "; ".join(
        ("%s s no valid judge" % w) if w in rk["no_valid"] else
        ("%s s best %s cons %.2f" % (w, label(rk["ranking"][w][0]["set"]), rk["ranking"][w][0]["consensus"]))
        for w in sorted(gw, key=float)))
    print("=" * W)


def per_clip_dump(var, ctl, meta):
    out = {}
    for base, by in var.items():
        ok = {w for r in RULES for w, c in ctl.get("%s|%s" % (base, r), {}).items() if c["usable"]}
        d = {}
        for name, x in by.items():
            if meta[name]["kind"] == "gen" and meta[name]["win"] not in ok:
                continue
            d[name] = {"score": [round(float(s), 5) for s in x["score"]], "top": [int(b) for b in x["hit"]]}
        out[base] = d
    return out


def parse(argv):
    ap = argparse.ArgumentParser(description="Validate water judges on real water drops, then score generated water.")
    ap.add_argument("--judges", nargs="+", default=["ast", "panns", "clap"])
    ap.add_argument("--tag-fits", nargs="+", default=TAG_FITS, choices=TAG_FITS)
    ap.add_argument("--clap-fits", nargs="+", default=CLAP_DEFAULT_FITS, choices=CLAP_FITS)
    ap.add_argument("--ctl-lengths", nargs="+", type=float, default=CTL_LENGTHS)
    ap.add_argument("--real-class", default=REAL_CLASS)
    ap.add_argument("--neg-classes", nargs="+", default=NEG_CLASSES)
    ap.add_argument("--noise-classes", nargs="+", default=NOISE_CLASSES)
    ap.add_argument("--probe-json", default=PROBE_JSON)
    ap.add_argument("--probe-dir", default=PROBE_DIR)
    ap.add_argument("--tier-dir", default=TIER_DIR)
    ap.add_argument("--ref", default=REF)
    ap.add_argument("--fpr", type=float, default=0.1)
    ap.add_argument("--min-real", type=float, default=0.5)
    ap.add_argument("--min-gap", type=float, default=0.3)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(ROOT, "scripts", "bench", "out_water_judge.json"))
    args = ap.parse_args(argv)
    bad = [j for j in args.judges if j not in JUDGES]
    if bad:
        raise SystemExit("unknown judges %s, choose from %s" % (bad, sorted(JUDGES)))
    return args


def main(argv=None):
    args = parse(argv)
    t0 = time.perf_counter()
    sets, meta = build_controls(args)
    gsets, gmeta, probe = build_generated(args)
    sets.update(gsets)
    meta.update(gmeta)
    print("sets (%d clips, %.1f s to load): %s" % (sum(len(v) for v in sets.values()), time.perf_counter() - t0,
                                                   ", ".join("%s=%d" % (label(k), len(v)) for k, v in sets.items())),
          flush=True)
    for g, m in gmeta.items():
        if not m["win"]:
            print("  %s: %.2f s clips have no control length within 0.05 s, left unjudged" % (g, m["dur"]))
    static = static_rates(sets, meta)
    print("static rule %s" % ("recomputed with water_probe.metrics" if probe_metrics else
                              "from out_water_probe.json only (%s)" % PROBE_IMPORT_ERROR), flush=True)
    var, labelsets = {}, {}
    for j in args.judges:
        t1 = time.perf_counter()
        res = JUDGES[j](sets, args)
        v, ls = clap_variants(res) if j == "clap" else tagger_variants(j, res)
        var.update(v)
        labelsets[j] = ls
        del res
        gc.collect()
        print("judged %-5s %.1f s" % (j, time.perf_counter() - t1), flush=True)
    ctl, taus = evaluate(var, args)
    gen = score_generated(var, ctl, taus, meta, args)
    rk = rank(gen, ctl, meta, static)
    report(sets, meta, var, ctl, gen, rk, static, labelsets, args)
    for by_g in gen.values():
        for rec in by_g.values():
            rec["pairs"] = [["%s#%02d" % (k, v), int(h)] for (k, v), h in rec["pairs"]]
    out = {"meta": {"root": ROOT, "args": vars(args), "sr": SR, "rules": list(RULES), "fits": {"tag": args.tag_fits,
                    "clap": args.clap_fits}, "diagnostic_sets": sorted(DIAGNOSTIC),
                    "clap_texts": {"water": CLAP_WATER, "diagnostic": CLAP_DIAG, "distractors": CLAP_DISTRACTORS},
                    "hiss": {"spectra_slope_highpass": HISS_SPECTRA, "per_spectrum": HISS_PER,
                             "rule": "usable = valid and real - hiss >= min_gap"},
                    "static_source": "recomputed" if probe_metrics else "probe_json",
                    "probe_import_error": PROBE_IMPORT_ERROR, "probe_json_found": probe is not None,
                    "seconds": time.perf_counter() - t0},
           "label_sets": labelsets,
           "sets": {k: {kk: vv for kk, vv in m.items() if kk != "pairs"} for k, m in meta.items()},
           "validity": ctl, "generated": gen, "selection": rk,
           "static": {k: {kk: vv for kk, vv in v.items()} for k, v in static.items()},
           "per_clip": per_clip_dump(var, ctl, meta)}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print("saved -> %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

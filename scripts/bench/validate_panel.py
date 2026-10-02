import _bootstrap

import argparse
import glob
import os

import numpy as np
import torch
import torchaudio.functional as AF

from sarge.generators import get_generator
from sarge.generators.stable_audio import _extract
from sarge.loudness import normalize_loudness, target_for
from sarge.types import AudioClip, CacheKey

CATEGORIES = ["a footstep", "a gunshot", "music", "speech", "silence", "rain", "an explosion"]
DISTRACTORS = ["hands clapping", "typing on a keyboard", "a hammer hitting", "a ball bouncing",
               "a clock ticking", "a door closing"]
CLAP_SETS = {"base": CATEGORIES, "hard": CATEGORIES + DISTRACTORS}
FOOT_TERMS = ("footstep", "walk")
TAGGERS = ("clap", "ast", "panns")
NEG_CLASSES = ("door_wood_knock", "glass_breaking", "water_drops", "crackling_fire", "wind")
AST_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
PANNS_SR = 32000
PANNS_WINDOW_S = 10.0


def _resample(clip, src, dst):
    if src == dst:
        return clip.astype(np.float32)
    t = torch.from_numpy(np.ascontiguousarray(clip)).float()
    return AF.resample(t, src, dst).numpy().astype(np.float32)


def fit_length(x, n, mode="none"):
    x = np.asarray(x, dtype=np.float32)
    if mode == "none" or len(x) == 0:
        return x
    if len(x) >= n:
        return x[:n]
    if mode == "silence":
        return np.pad(x, (0, n - len(x)))
    return np.tile(x, n // len(x) + 1)[:n]


def _foot_ids(names):
    return [int(i) for i, n in names.items() if any(t in n.lower() for t in FOOT_TERMS)]


def _mean(rows, key):
    v = [float(r[key]) for r in rows if r.get(key) is not None]
    return float(np.mean(v)) if v else float("nan")


def summarize(rows):
    out = {}
    if not rows:
        return out
    if "foot" in rows[0]:
        out["footstep_rate"] = _mean(rows, "foot")
        out["p_foot"] = float(np.median([r["p_foot"] for r in rows]))
    if "mat" in rows[0]:
        out["material_acc"] = _mean(rows, "mat")
    if "top1" in rows[0]:
        out["top1"] = [r["top1"] for r in rows[:6]]
    for k in ("CE", "CU", "PC", "PQ"):
        if k in rows[0]:
            out[k] = _mean(rows, k)
    return out


def judge_clap(clips_48k, labels, materials, dev, cats=None, bs=32):
    from transformers import ClapModel, ClapProcessor
    cats = list(cats or CATEGORIES)
    m = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(dev).eval()
    pr = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")

    def classify(texts):
        t = pr(text=texts, return_tensors="pt", padding=True)
        out = []
        for i in range(0, len(clips_48k), bs):
            a = pr(audio=clips_48k[i: i + bs], sampling_rate=48000, return_tensors="pt")
            inp = {k: v.to(dev) for k, v in {**a, **t}.items()}
            with torch.no_grad():
                out.append(m(**inp).logits_per_audio.float().cpu().numpy())
        return np.concatenate(out)

    cat = classify(cats)
    fi = cats.index("a footstep")
    p = np.exp(cat - cat.max(1, keepdims=True))
    p = p / p.sum(1, keepdims=True)
    known = [l in materials for l in labels]
    mat = classify([f"the sound of a footstep on {x}" for x in materials]).argmax(1) if any(known) else None
    rows = [{"foot": bool(cat[i].argmax() == fi), "p_foot": float(p[i, fi]),
             "mat": bool(mat[i] == materials.index(l)) if k else None}
            for i, (l, k) in enumerate(zip(labels, known))]
    del m
    torch.cuda.empty_cache()
    return {**summarize(rows), "rows": rows}


def judge_ast(clips_16k, dev, fit="none", bs=8):
    from transformers import ASTForAudioClassification, AutoFeatureExtractor
    m = ASTForAudioClassification.from_pretrained(AST_NAME).to(dev).eval()
    fe = AutoFeatureExtractor.from_pretrained(AST_NAME)
    id2label = m.config.id2label
    foot = _foot_ids(id2label)
    n = 400 + (fe.max_length - 1) * 160
    rows = []
    for i in range(0, len(clips_16k), bs):
        xs = [fit_length(c, n, fit) for c in clips_16k[i: i + bs]]
        inp = fe(xs, sampling_rate=16000, return_tensors="pt")
        inp = {k: v.to(dev) for k, v in inp.items()}
        with torch.no_grad():
            lg = m(**inp).logits.float().cpu()
        for row, p in zip(lg, torch.sigmoid(lg)):
            idx = torch.topk(row, 5).indices.tolist()
            rows.append({"foot": any(j in foot for j in idx), "p_foot": float(p[foot].max()),
                         "top1": id2label[idx[0]].lower()})
    del m
    torch.cuda.empty_cache()
    return {**summarize(rows), "rows": rows}


def judge_panns(clips_32k, dev, fit="none", bs=16):
    from panns_inference import AudioTagging
    from panns_inference.config import labels
    at = AudioTagging(checkpoint_path=None, device=dev)
    foot = _foot_ids(dict(enumerate(labels)))
    n = int(PANNS_WINDOW_S * PANNS_SR)
    xs = [fit_length(c, n, fit) for c in clips_32k]
    rows = []
    i = 0
    while i < len(xs):
        j = i + 1
        while j < len(xs) and j - i < bs and len(xs[j]) == len(xs[i]):
            j += 1
        clipwise, _ = at.inference(np.stack(xs[i:j]).astype(np.float32))
        for row in clipwise:
            idx = [int(k) for k in np.argsort(row)[::-1][:5]]
            rows.append({"foot": any(k in foot for k in idx), "p_foot": float(row[foot].max()),
                         "top1": labels[idx[0]].lower()})
        i = j
    del at
    torch.cuda.empty_cache()
    return {**summarize(rows), "rows": rows}


def judge_audiobox(clips, sr=None, bs=16):
    from audiobox_aesthetics.infer import initialize_predictor
    p = initialize_predictor()
    rows = []
    for i in range(0, len(clips), bs):
        rows += p.forward([{"path": c} if isinstance(c, str) else
                           {"path": torch.from_numpy(np.ascontiguousarray(c, dtype=np.float32))[None],
                            "sample_rate": int(sr)}
                           for c in clips[i: i + bs]])
    del p
    torch.cuda.empty_cache()
    return {**summarize(rows), "rows": rows}


def load_set(paths, sr, dur, intensity="normal", keep_s=0.0):
    out = []
    for pth in paths:
        c = AudioClip.from_wav(pth)
        x = _extract(_resample(c.samples, c.sample_rate, sr), sr, dur)
        if keep_s > 0:
            a, b = int(keep_s * sr), int((keep_s + 0.05) * sr)
            x = x.copy()
            x[a:b] *= np.linspace(1.0, 0.0, len(x[a:b]), dtype=np.float32)
            x[b:] = 0.0
        out.append(normalize_loudness(x, target_rms=target_for(intensity)))
    return out


def neg_paths(root, classes=NEG_CLASSES):
    return sorted(p for c in classes for p in glob.glob(os.path.join(root, c, "*.wav")))


def run_judges(sets, labels, materials, judges, sr, dev, fit="tile", cats=None):
    names = list(sets)
    flat = [x for n in names for x in sets[n]]
    owner = [n for n in names for _ in sets[n]]
    labs = [l for n in names for l in labels.get(n, [None] * len(sets[n]))]
    res = {n: {} for n in names}
    for j in judges:
        idx = [i for i, o in enumerate(owner) if j != "audiobox" or not o.startswith("neg")]
        xs = [flat[i] for i in idx]
        if not xs:
            continue
        if j == "clap":
            r = judge_clap([_resample(x, sr, 48000) for x in xs], [labs[i] for i in idx], materials, dev, cats)
        elif j == "ast":
            r = judge_ast([_resample(x, sr, 16000) for x in xs], dev, fit)
        elif j == "panns":
            r = judge_panns([_resample(x, sr, PANNS_SR) for x in xs], dev, fit)
        elif j == "audiobox":
            r = judge_audiobox(xs, sr)
        else:
            continue
        for n in names:
            rows = [row for row, i in zip(r["rows"], idx) if owner[i] == n]
            if rows:
                res[n][j] = summarize(rows)
        print("judged %s" % j)
    return res


def anchor(gen, real, neg=None, min_real=0.5, min_gap=0.3):
    out = {}
    for j in TAGGERS:
        if j not in gen or j not in real:
            continue
        g, rr = gen[j]["footstep_rate"], real[j]["footstep_rate"]
        nr = neg[j]["footstep_rate"] if neg and j in neg else float("nan")
        out[j] = {"gen": g, "real": rr, "neg": nr, "ratio": g / rr if rr > 0 else float("nan"),
                  "valid": bool(rr >= min_real and (np.isnan(nr) or rr - nr >= min_gap))}
    ok = [min(v["ratio"], 1.0) for v in out.values() if v["valid"]]
    return out, (min(ok) if ok else float("nan"))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Panel of independent models validating generated SFX against real audio.")
    p.add_argument("--generator", default="sa3")
    p.add_argument("--materials", nargs="+", default=["wood", "stone", "grass", "metal"])
    p.add_argument("--intensity", default="normal")
    p.add_argument("--variants", type=int, default=10)
    p.add_argument("--judges", nargs="+", default=["clap", "ast", "panns", "audiobox"])
    p.add_argument("--fit", default="tile", choices=["tile", "silence", "none"])
    p.add_argument("--clap-set", default="hard", choices=sorted(CLAP_SETS))
    p.add_argument("--real", default=os.path.join("data", "real_footsteps"))
    p.add_argument("--neg", default=os.path.join("data", "esc50"))
    p.add_argument("--real-keep", type=float, default=0.0)
    p.add_argument("--min-real", type=float, default=0.5)
    p.add_argument("--min-gap", type=float, default=0.3)
    args = p.parse_args(argv)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = get_generator(args.generator)
    clips, labels = [], []
    tmp = os.path.join("data", "_panel")
    os.makedirs(tmp, exist_ok=True)
    for mat in args.materials:
        for v in range(args.variants):
            clip = g.generate(CacheKey("footstep", mat, args.intensity), variant=v)
            clips.append(clip.samples)
            labels.append(mat)
            clip.to_wav(os.path.join(tmp, f"{mat}_{v:02d}.wav"))
    sr = int(g.sample_rate)
    dur = float(getattr(g, "duration_s", len(clips[0]) / sr))
    del g
    torch.cuda.empty_cache()

    sets = {"gen": clips,
            "real": load_set(sorted(glob.glob(os.path.join(args.real, "*.wav"))), sr, dur, args.intensity,
                             args.real_keep)}
    negs = neg_paths(args.neg)
    if negs:
        sets["neg"] = load_set(negs, sr, dur, args.intensity)
    res = run_judges(sets, {"gen": labels}, args.materials, args.judges, sr, dev, args.fit,
                     CLAP_SETS[args.clap_set])
    rel, cons = anchor(res["gen"], res["real"], res.get("neg"), args.min_real, args.min_gap)

    chance = 100.0 / len(args.materials)
    print(f"\nclips: {len(clips)} generated ({args.variants} x {len(args.materials)} materials), "
          f"{len(sets['real'])} real crops, {len(sets.get('neg', []))} negative crops, "
          f"fit={args.fit}, clap={args.clap_set}\n")
    print("=" * 72)
    print(" AUTOMATED VALIDATION PANEL   footstep rate on %.2f s clips" % dur)
    print("=" * 72)
    print(" %-8s %7s %7s %7s %9s  %s" % ("judge", "real", "neg", "gen", "gen/real", "valid"))
    for j, v in rel.items():
        print(" %-8s %6.1f%% %6.1f%% %6.1f%% %9.2f  %s" % (
            j, 100 * v["real"], 100 * v["neg"], 100 * v["gen"], v["ratio"], "yes" if v["valid"] else "NO"))
    if "clap" in res["gen"]:
        print(f" CLAP material {100 * res['gen']['clap']['material_acc']:5.1f}% (chance {chance:.0f}%)")
    for j in ("ast", "panns"):
        if j in res["gen"]:
            print(f" {j.upper():5s} top1 gen {res['gen'][j]['top1'][:3]}  real {res['real'].get(j, {}).get('top1', [])[:3]}")
    for name in ("gen", "real"):
        a = res[name].get("audiobox")
        if a:
            print(f" Audiobox {name:4s} " + "  ".join(f"{k} {a[k]:.2f}" for k in ("CE", "CU", "PC", "PQ")) + "   (scale 1-10)")
    print("-" * 72)
    print(f" RELATIVE CONSENSUS (min gen/real over valid taggers): {cons:.2f}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

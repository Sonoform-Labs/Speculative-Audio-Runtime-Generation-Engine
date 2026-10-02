import _bootstrap

import argparse
import glob
import json
import os
import time

import numpy as np
import torch
from scipy.io import wavfile

from sarge.generators import get_generator
from sarge.types import CacheKey
from validate_panel import _resample, CLAP_SETS, TAGGERS, anchor, load_set, neg_paths, run_judges


def onset_window(x, sr, dur=0.3):
    env = np.abs(x)
    if env.max() <= 0:
        return x[: int(dur * sr)]
    i = int(np.argmax(env > 0.1 * env.max()))
    return x[i: i + int(dur * sr)]


def crest_db(x):
    rms = float(np.sqrt(np.mean(x ** 2))) + 1e-12
    return float(20 * np.log10(np.max(np.abs(x)) / rms + 1e-12))


def attack_ms(x, sr):
    w = max(1, int(0.001 * sr))
    env = np.convolve(np.abs(x), np.ones(w) / w, mode="same")
    pk = int(np.argmax(env))
    p = env[pk]
    if p <= 0:
        return 0.0
    a = int(np.argmax(env[: pk + 1] >= 0.1 * p))
    b = int(np.argmax(env[: pk + 1] >= 0.9 * p))
    return (b - a) / sr * 1000.0


def load_wav(p):
    sr, a = wavfile.read(p)
    a = a.astype(np.float32)
    if a.ndim > 1:
        a = a.mean(1)
    if np.max(np.abs(a)) > 1.5:
        a = a / 32768.0
    return a, sr


def transients(clips, sr):
    c, a = [], []
    for x in clips:
        w = onset_window(x, sr)
        if len(w) < int(0.02 * sr):
            continue
        c.append(crest_db(w))
        a.append(attack_ms(w, sr))
    return {"crest_db": float(np.mean(c)), "crest_sd": float(np.std(c)),
            "attack_ms": float(np.median(a)), "n": len(c)}


def verdict(r, real_j, neg_j, real_t, args):
    rel, cons = anchor(r, real_j, neg_j, args.min_real, args.min_gap)
    t = r["transients"]
    checks = {"taggers": bool(cons >= args.k),
              "crest": bool(t["crest_db"] >= real_t["crest_db"] - args.crest_sd * real_t["crest_sd"])}
    if "audiobox" in r and "audiobox" in real_j:
        checks["pq"] = bool(r["audiobox"]["PQ"] >= real_j["audiobox"]["PQ"] - args.pq_margin)
    return {"consensus": cons, "judges": rel, "checks": checks, "pass": all(checks.values())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="4,3,2,1")
    ap.add_argument("--variants", type=int, default=10)
    ap.add_argument("--materials", nargs="+", default=["wood", "stone", "grass", "metal"])
    ap.add_argument("--judges", nargs="+", default=["clap", "ast", "panns", "audiobox"])
    ap.add_argument("--real", default=os.path.join("data", "real_footsteps"))
    ap.add_argument("--neg", default=os.path.join("data", "esc50"))
    ap.add_argument("--real-keep", type=float, default=0.0)
    ap.add_argument("--out", default=os.path.join("scripts", "bench", "out_gate.json"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--mid-noise", type=float, default=None)
    ap.add_argument("--padding", type=float, default=None)
    ap.add_argument("--fit", default="tile", choices=["tile", "silence", "none"])
    ap.add_argument("--clap-set", default="hard", choices=sorted(CLAP_SETS))
    ap.add_argument("--k", type=float, default=0.8)
    ap.add_argument("--crest-sd", type=float, default=1.0)
    ap.add_argument("--pq-margin", type=float, default=0.5)
    ap.add_argument("--min-real", type=float, default=0.5)
    ap.add_argument("--min-gap", type=float, default=0.3)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    steps_list = [int(s) for s in args.steps.split(",")]

    gen = get_generator("sa3")
    gen.mid_noise = args.mid_noise
    if args.padding is not None:
        gen.duration_padding_s = args.padding
    gen.ensure_ready()
    sr = int(gen.sample_rate)
    for k in (CacheKey("footstep", "wood", "normal"), CacheKey("impact", "glass", "normal")):
        gen.generate(k, variant=0)

    configs = {}
    for s in steps_list:
        gen.steps = s
        tag = "steps%d_b%d%s" % (s, args.batch, args.tag)
        d = os.path.join("data", "_gate", tag)
        os.makedirs(d, exist_ok=True)
        clips, labels, paths, ms = [], [], [], []
        items = [(CacheKey("footstep", mat, "normal"), v)
                 for mat in args.materials for v in range(args.variants)]
        for i in range(0, len(items), args.batch):
            chunk = items[i: i + args.batch]
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            made = (gen.generate_batch(chunk) if args.batch > 1
                    else [gen.generate(k, variant=v) for k, v in chunk])
            torch.cuda.synchronize()
            per = (time.perf_counter() - t0) * 1000.0 / len(chunk)
            for (k, v), clip in zip(chunk, made):
                ms.append(per)
                pth = os.path.join(d, "%s_%02d.wav" % (k.material, v))
                clip.to_wav(pth)
                clips.append(clip.samples); labels.append(k.material); paths.append(pth)
        configs[tag] = {"steps": s, "clips": clips, "labels": labels, "paths": paths,
                        "gen_ms": float(np.median(ms))}
        print("generated %-10s %3d clips  median %.1f ms" % (tag, len(clips), np.median(ms)))

    dur = float(gen.duration_s)
    del gen
    torch.cuda.empty_cache()

    real_paths = sorted(glob.glob(os.path.join(args.real, "*.wav")))
    real = [load_wav(p) for p in real_paths]
    real_t = transients([_resample(x, r, sr) for x, r in real], sr)

    sets = {"real_crop": load_set(real_paths, sr, dur, keep_s=args.real_keep)}
    negs = neg_paths(args.neg)
    if negs:
        sets["neg"] = load_set(negs, sr, dur)
    sets.update({tag: c["clips"] for tag, c in configs.items()})
    judged = run_judges(sets, {tag: c["labels"] for tag, c in configs.items()}, args.materials,
                        args.judges, sr, dev, args.fit, CLAP_SETS[args.clap_set])

    res = {"real_footsteps": {"transients": real_t}, "real_crop": judged["real_crop"],
           "neg": judged.get("neg", {})}
    for tag, c in configs.items():
        res[tag] = {"steps": c["steps"], "gen_ms": c["gen_ms"],
                    "transients": transients(c["clips"], sr), **judged[tag]}
        res[tag]["gate"] = verdict(res[tag], res["real_crop"], res["neg"] or None, real_t, args)

    rt = real_t
    rc, ng = res["real_crop"], res["neg"]
    nan = float("nan")
    print()
    print("=" * 122)
    print(" SPEED + QUALITY GATE   (real footsteps: crest %.1f dB, attack %.2f ms, n=%d; fit=%s, clap=%s)"
          % (rt["crest_db"], rt["attack_ms"], rt["n"], args.fit, args.clap_set))
    print(" real %.2f s crops vs negatives:  " % dur + "   ".join(
        "%s %.0f%% / %.0f%%" % (j, 100 * rc[j]["footstep_rate"], 100 * ng.get(j, {}).get("footstep_rate", nan))
        for j in TAGGERS if j in rc) + ("   real crop PQ %.2f" % rc["audiobox"]["PQ"] if "audiobox" in rc else ""))
    print("=" * 122)
    hdr = "%-10s %7s %8s %9s %8s %8s %6s %6s %6s %6s %6s %6s %6s %6s"
    print(hdr % ("config", "gen_ms", "crest", "attack", "clapFt", "clapMat", "ast", "panns",
                 "CE", "CU", "PC", "PQ", "cons", "gate"))
    print("-" * 122)
    for tag, c in configs.items():
        r = res[tag]
        t = r["transients"]
        ab = r.get("audiobox", {})
        gt = r["gate"]
        print("%-10s %7.1f %8.1f %9.2f %7.0f%% %7.0f%% %5.0f%% %5.0f%% %6.2f %6.2f %6.2f %6.2f %6.2f %6s" % (
            tag, r["gen_ms"], t["crest_db"], t["attack_ms"],
            100 * r.get("clap", {}).get("footstep_rate", nan),
            100 * r.get("clap", {}).get("material_acc", nan),
            100 * r.get("ast", {}).get("footstep_rate", nan),
            100 * r.get("panns", {}).get("footstep_rate", nan),
            ab.get("CE", nan), ab.get("CU", nan), ab.get("PC", nan), ab.get("PQ", nan),
            gt["consensus"], "PASS" if gt["pass"] else "FAIL"))
        bad = [k for k, v in gt["checks"].items() if not v]
        skip = [j for j, v in gt["judges"].items() if not v["valid"]]
        if bad or skip:
            print("%-10s   failed: %s   invalid judges: %s" % ("", ", ".join(bad) or "-", ", ".join(skip) or "-"))
    print("-" * 122)
    print(" crest = peak/RMS in dB over the first 300 ms, higher = sharper transient")
    print(" attack = 10->90 percent rise of the 1 ms envelope, reported only")
    print(" cons = min over valid taggers of gen rate / real crop rate; valid = real >= %.2f and real - neg >= %.2f"
          % (args.min_real, args.min_gap))
    print(" gate = cons >= %.2f, crest >= real - %.1f sd, PQ >= real crop PQ - %.2f"
          % (args.k, args.crest_sd, args.pq_margin))
    print("=" * 122)

    out = {k: v for k, v in res.items()}
    out["gate_args"] = {k: getattr(args, k) for k in ("fit", "clap_set", "k", "crest_sd", "pq_margin",
                                                        "min_real", "min_gap", "variants", "real_keep")}
    json.dump(out, open(args.out, "w"), indent=1)
    print("saved -> %s" % args.out)


if __name__ == "__main__":
    main()

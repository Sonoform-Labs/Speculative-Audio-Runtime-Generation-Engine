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
from validate_panel import _resample, judge_clap, judge_ast, judge_panns, judge_audiobox


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="4,3,2,1")
    ap.add_argument("--variants", type=int, default=5)
    ap.add_argument("--materials", nargs="+", default=["wood", "stone", "grass", "metal"])
    ap.add_argument("--judges", nargs="+", default=["clap", "ast", "panns", "audiobox"])
    ap.add_argument("--real", default=os.path.join("data", "real_footsteps"))
    ap.add_argument("--out", default=os.path.join("scripts", "bench", "out_gate.json"))
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    steps_list = [int(s) for s in args.steps.split(",")]

    gen = get_generator("sa3")
    gen.ensure_ready()
    sr = int(gen.sample_rate)
    for k in (CacheKey("footstep", "wood", "normal"), CacheKey("impact", "glass", "normal")):
        gen.generate(k, variant=0)

    configs = {}
    for s in steps_list:
        gen.steps = s
        tag = "steps%d%s" % (s, args.tag)
        d = os.path.join("data", "_gate", tag)
        os.makedirs(d, exist_ok=True)
        clips, labels, paths, ms = [], [], [], []
        for mat in args.materials:
            for v in range(args.variants):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                clip = gen.generate(CacheKey("footstep", mat, "normal"), variant=v)
                torch.cuda.synchronize()
                ms.append((time.perf_counter() - t0) * 1000.0)
                pth = os.path.join(d, "%s_%02d.wav" % (mat, v))
                clip.to_wav(pth)
                clips.append(clip.samples); labels.append(mat); paths.append(pth)
        configs[tag] = {"steps": s, "clips": clips, "labels": labels, "paths": paths,
                        "gen_ms": float(np.median(ms))}
        print("generated %-10s %3d clips  median %.1f ms" % (tag, len(clips), np.median(ms)))

    del gen
    torch.cuda.empty_cache()

    real = [load_wav(p) for p in sorted(glob.glob(os.path.join(args.real, "*.wav")))]
    real_t = transients([_resample(x, r, sr) for x, r in real], sr)

    res = {"real_footsteps": {"transients": real_t}}
    for tag, c in configs.items():
        res[tag] = {"steps": c["steps"], "gen_ms": c["gen_ms"],
                    "transients": transients(c["clips"], sr)}

    for j in args.judges:
        for tag, c in configs.items():
            if j == "clap":
                r = judge_clap([_resample(x, sr, 48000) for x in c["clips"]], c["labels"],
                               args.materials, dev)
            elif j == "ast":
                r = judge_ast([_resample(x, sr, 16000) for x in c["clips"]], dev)
            elif j == "panns":
                r = judge_panns([_resample(x, sr, 32000) for x in c["clips"]], dev)
            elif j == "audiobox":
                r = judge_audiobox(c["paths"])
            else:
                continue
            res[tag][j] = {k: v for k, v in r.items() if k != "top1"}
        print("judged %s" % j)

    rt = real_t
    print()
    print("=" * 108)
    print(" SPEED + QUALITY GATE   (real footsteps: crest %.1f dB, attack %.2f ms, n=%d)"
          % (rt["crest_db"], rt["attack_ms"], rt["n"]))
    print("=" * 108)
    hdr = "%-10s %7s %8s %9s %8s %8s %6s %6s %6s %6s %6s %6s"
    print(hdr % ("config", "gen_ms", "crest", "attack", "clapFt", "clapMat", "ast", "panns",
                 "CE", "CU", "PC", "PQ"))
    print("-" * 108)
    for tag, c in configs.items():
        r = res[tag]
        t = r["transients"]
        ab = r.get("audiobox", {})
        print("%-10s %7.1f %8.1f %9.2f %7.0f%% %7.0f%% %5.0f%% %5.0f%% %6.2f %6.2f %6.2f %6.2f" % (
            tag, r["gen_ms"], t["crest_db"], t["attack_ms"],
            100 * r.get("clap", {}).get("footstep_rate", float("nan")),
            100 * r.get("clap", {}).get("material_acc", float("nan")),
            100 * r.get("ast", {}).get("footstep_rate", float("nan")),
            100 * r.get("panns", {}).get("footstep_rate", float("nan")),
            ab.get("CE", float("nan")), ab.get("CU", float("nan")),
            ab.get("PC", float("nan")), ab.get("PQ", float("nan"))))
    print("-" * 108)
    print(" crest = peak/RMS in dB over the first 300 ms, higher = sharper transient")
    print(" attack = 10->90 percent rise of the 1 ms envelope; closer to real = more realistic")
    print("=" * 108)

    out = {k: v for k, v in res.items()}
    json.dump(out, open(args.out, "w"), indent=1)
    print("saved -> %s" % args.out)


if __name__ == "__main__":
    main()

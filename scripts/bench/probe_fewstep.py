import sys

sys.path.insert(0, r"D:\Python\SAGE\scripts\bench")

import _bootstrap

import argparse
import glob
import json
import os
import time

import numpy as np
import torch

from stable_audio_3.inference.distribution_shift import FluxDistributionShift
from sarge.generators.stable_audio import _extract, _seed
from sarge.generators.stable_audio3 import StableAudio3Generator
from sarge.loudness import normalize_loudness, target_for
from sarge.types import AudioClip, CacheKey
from gate_quality import load_wav, transients
from validate_panel import _resample, judge_audiobox, judge_clap

HERE = os.path.dirname(os.path.abspath(__file__))
REAL = r"D:\Python\SAGE\data\real_footsteps"

CONFIGS = {
    "s3_euler_cfg2": (3, "euler", 2.0, None, 6.0),
    "s3_euler_cfg1": (3, "euler", 1.0, None, 6.0),
    "s3_euler_cfg2_pad1": (3, "euler", 2.0, None, 1.0),
    "s2_euler_cfg2": (2, "euler", 2.0, None, 6.0),
    "s2_euler_cfg2_m50": (2, "euler", 2.0, 0.5, 6.0),
    "s2_euler_cfg2_m60": (2, "euler", 2.0, 0.6, 6.0),
    "s2_euler_cfg2_m68": (2, "euler", 2.0, 0.68, 6.0),
    "s2_euler_cfg2_m80": (2, "euler", 2.0, 0.8, 6.0),
    "s2_euler_cfg1_m68": (2, "euler", 1.0, 0.68, 6.0),
    "s2_pingpong_cfg1": (2, "pingpong", 1.0, None, 6.0),
    "s2_pingpong_cfg1_m68": (2, "pingpong", 1.0, 0.68, 6.0),
    "s2_pingpong_cfg2_m68": (2, "pingpong", 2.0, 0.68, 6.0),
}


def mid_shift(m):
    a = m / (1.0 - m)
    return FluxDistributionShift(alpha_min=a, alpha_max=a)


def make(gen, key, variant, cfg):
    steps, sampler, scale, m, pad = cfg
    kw = {"sampler_type": sampler, "disable_tqdm": True, "duration_padding_sec": pad}
    if m is not None:
        kw["dist_shift"] = mid_shift(m)
    with torch.no_grad():
        audio = gen._model.generate(prompt=gen._prompt(key), duration=gen.gen_duration_s,
                                    steps=steps, cfg_scale=scale, seed=_seed(key, variant), **kw)
    a = audio.detach().to(torch.float32).cpu().numpy()
    while a.ndim > 2:
        a = a[0]
    mono = a.mean(axis=0) if a.ndim == 2 else a
    mono = _extract(mono, gen.sample_rate, gen.duration_s)
    mono = normalize_loudness(mono, target_rms=target_for(key.intensity))
    return AudioClip(mono.astype(np.float32), gen.sample_rate, variant_id=variant)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--variants", type=int, default=5)
    ap.add_argument("--materials", nargs="+", default=["wood", "stone", "grass", "metal"])
    ap.add_argument("--judges", nargs="+", default=["clap", "audiobox"])
    ap.add_argument("--out", default=os.path.join(HERE, "out_fewstep.json"))
    ap.add_argument("--wavs", default=os.path.join(HERE, "wavs"))
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    names = [n for n in args.configs.split(",") if n]
    gen = StableAudio3Generator()
    gen.ensure_ready()
    sr = int(gen.sample_rate)
    for k in (CacheKey("footstep", "wood", "normal"), CacheKey("impact", "glass", "normal")):
        gen.generate(k, variant=0)

    items = [(CacheKey("footstep", mat, "normal"), v)
             for mat in args.materials for v in range(args.variants)]
    runs = {}
    for name in names:
        cfg = CONFIGS[name]
        d = os.path.join(args.wavs, name)
        os.makedirs(d, exist_ok=True)
        clips, labels, paths, ms = [], [], [], []
        make(gen, items[0][0], 99, cfg)
        for k, v in items:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            clip = make(gen, k, v, cfg)
            torch.cuda.synchronize()
            ms.append((time.perf_counter() - t0) * 1000.0)
            p = os.path.join(d, "%s_%02d.wav" % (k.material, v))
            clip.to_wav(p)
            clips.append(clip.samples)
            labels.append(k.material)
            paths.append(p)
        runs[name] = {"cfg": list(cfg), "clips": clips, "labels": labels, "paths": paths,
                      "gen_ms": float(np.median(ms))}
        print("generated %-24s median %.1f ms" % (name, np.median(ms)))

    del gen
    torch.cuda.empty_cache()

    real = [load_wav(p) for p in sorted(glob.glob(os.path.join(REAL, "*.wav")))]
    res = {"real_footsteps": transients([_resample(x, r, sr) for x, r in real], sr)}
    for name, r in runs.items():
        res[name] = {"cfg": r["cfg"], "gen_ms": r["gen_ms"], "transients": transients(r["clips"], sr)}

    def save():
        json.dump(res, open(args.out, "w"), indent=1)

    save()
    for j in args.judges:
        for name, r in runs.items():
            if j == "clap":
                out = judge_clap([_resample(x, sr, 48000) for x in r["clips"]], r["labels"],
                                 args.materials, dev)
            elif j == "audiobox":
                out = judge_audiobox(r["paths"])
            else:
                continue
            res[name][j] = {k: v for k, v in out.items() if k != "top1"}
        save()
        print("judged %s" % j)

    rt = res["real_footsteps"]
    print()
    print("real footsteps: crest %.1f dB, attack %.2f ms, n=%d" % (rt["crest_db"], rt["attack_ms"], rt["n"]))
    hdr = "%-24s %7s %7s %8s %7s %7s %6s"
    print(hdr % ("config", "gen_ms", "crest", "attack", "clapFt", "clapMat", "PQ"))
    print("-" * 72)
    for name in runs:
        r = res[name]
        t = r["transients"]
        c = r.get("clap", {})
        print("%-24s %7.1f %7.1f %8.2f %6.0f%% %6.0f%% %6.2f" % (
            name, r["gen_ms"], t["crest_db"], t["attack_ms"],
            100 * c.get("footstep_rate", float("nan")), 100 * c.get("material_acc", float("nan")),
            r.get("audiobox", {}).get("PQ", float("nan"))))
    save()
    print("saved -> %s" % args.out)


if __name__ == "__main__":
    main()

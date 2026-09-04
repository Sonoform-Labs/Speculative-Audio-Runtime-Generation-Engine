import _bootstrap

import argparse
import csv
import json
import os
import time
import traceback

import numpy as np
import torch

from sarge.generators.stable_audio3 import StableAudio3Generator
from sarge.generators.stable_audio import _extract
from sarge.loudness import normalize_loudness, target_for
from sarge.types import AudioClip

PROMPTS = [
    ("footstep_wood",   "a single footstep on a wooden floor, dry, isolated sound effect"),
    ("footstep_metal",  "a single footstep on a metal floor, dry, isolated sound effect"),
    ("footstep_gravel", "a single footstep on gravel, dry, isolated sound effect"),
    ("footstep_grass",  "a single footstep on grass, dry, isolated sound effect"),
    ("glass_impact",    "a glass impact, dry, isolated sound effect"),
    ("glass_shatter",   "glass shattering, dry, isolated sound effect"),
    ("metal_hit",       "a metal hit, dry, isolated sound effect"),
    ("metal_scrape",    "metal scraping, dry, isolated sound effect"),
    ("door_creak",      "a door creaking, dry, isolated sound effect"),
    ("door_slam",       "a door slamming shut, dry, isolated sound effect"),
    ("plastic_crack",   "a plastic crack snapping, dry, isolated sound effect"),
    ("water_splash",    "a water splash, dry, isolated sound effect"),
    ("cloth_rustle",    "cloth rustling, dry, isolated sound effect"),
    ("heavy_landing",   "a heavy object landing on the ground, dry, isolated sound effect"),
]

CONFIGS = (
    [("euler", s) for s in (8, 4, 3, 2, 1)]
    + [("dpmpp", s) for s in (4, 3, 2)]
    + [("rk4", s) for s in (4, 3, 2)]
    + [("pingpong", s) for s in (4, 3, 2)]
)


def gen_one(g, prompt, steps, sampler_type, cfg, seed, gen_dur, out_dur):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        audio = g._model.generate(
            prompt=prompt, duration=gen_dur, steps=steps,
            cfg_scale=cfg, seed=seed, sampler_type=sampler_type, disable_tqdm=True)
    torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - t0) * 1000.0
    a = audio.detach().to(torch.float32).cpu().numpy()
    while a.ndim > 2:
        a = a[0]
    mono = a.mean(axis=0) if a.ndim == 2 else a
    sr = int(g.sample_rate)
    mono = _extract(mono, sr, out_dur)
    mono = normalize_loudness(mono, target_rms=target_for("normal"))
    return mono.astype(np.float32), sr, latency_ms


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/sampler-bench")
    ap.add_argument("--cfg", type=float, default=2.0)
    ap.add_argument("--gen-dur", type=float, default=1.0)
    ap.add_argument("--out-dur", type=float, default=0.6)
    ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument("--warm", type=int, default=2)
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    g = StableAudio3Generator()
    g.ensure_ready()
    sr = int(g.sample_rate)

    settings = dict(cfg=args.cfg, gen_dur=args.gen_dur, out_dur=args.out_dur,
                    seed_base=args.seed_base, sample_rate=sr,
                    model="stable-audio-3-small-sfx", device=str(g.device),
                    n_prompts=len(PROMPTS), configs=[f"{s}@{n}" for s, n in CONFIGS])
    with open(os.path.join(args.out, "settings.json"), "w") as f:
        json.dump(settings, f, indent=2)
    print("settings:", json.dumps(settings))

    for s_type, _ in {(s, 0) for s, _ in CONFIGS}:
        try:
            gen_one(g, "warmup", 2, s_type, args.cfg, 0, args.gen_dur, args.out_dur)
        except Exception:
            pass

    rows = []
    for pi, (slug, prompt) in enumerate(PROMPTS):
        seed = args.seed_base + pi
        for sampler, steps in CONFIGS:
            cfgdir = os.path.join(args.out, f"{sampler}_{steps}step")
            os.makedirs(cfgdir, exist_ok=True)
            wav = os.path.join(cfgdir, f"{slug}_seed{seed}.wav")
            row = dict(slug=slug, prompt=prompt, seed=seed, sampler=sampler, steps=steps,
                       cfg=args.cfg, gen_dur=args.gen_dur, out_dur=args.out_dur,
                       sample_rate=sr, wav=os.path.relpath(wav, args.out), error="")
            try:
                mono, sr, lat = gen_one(g, prompt, steps, sampler, args.cfg, seed,
                                        args.gen_dur, args.out_dur)
                AudioClip(mono, sr).to_wav(wav)
                row["latency_ms"] = round(lat, 1)
                row["rms"] = float(np.sqrt(np.mean(mono ** 2)))
                row["peak"] = float(np.max(np.abs(mono)))
                row["n_samples"] = int(mono.shape[0])
            except Exception as e:
                row["latency_ms"] = ""
                row["error"] = f"{type(e).__name__}: {e}"
                traceback.print_exc()
            rows.append(row)
            print(f"{sampler:8s} {steps}step  {slug:16s} seed={seed}  "
                  f"lat={row.get('latency_ms','ERR')}ms  {row['error']}")

    cols = ["slug", "prompt", "seed", "sampler", "steps", "cfg", "gen_dur", "out_dur",
            "sample_rate", "latency_ms", "rms", "peak", "n_samples", "wav", "error"]
    man = os.path.join(args.out, "manifest.csv")
    with open(man, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    ok = sum(1 for r in rows if not r["error"])
    print(f"\nDONE: {ok}/{len(rows)} ok -> {man}")


if __name__ == "__main__":
    main()

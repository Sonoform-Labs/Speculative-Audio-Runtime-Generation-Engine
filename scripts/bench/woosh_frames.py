import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent / "src"))

import numpy as np

import woosh_server as ws
from sarge.prompts import build
from sarge.types import AudioClip, CacheKey, seed_for

OUT = HERE.parent.parent / "data" / "_gate" / "woosh_frames"
KEYS = [("break", "window"), ("footstep", "water"), ("splash", "water"), ("footstep", "gravel"), ("footstep", "stone"),
        ("bullet impact", "glass")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(ws.MODELS), required=True)
    ap.add_argument("--frames", default="101,151,501")
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    import torch
    rep = {"model": a.model, "rows": []}
    cache = {}
    for fr in [int(f) for f in a.frames.split(",")]:
        g = gen_for(a.model, fr, cache)
        g("warmup, a soft thud", 0)
        for et, mat in KEYS:
            k = CacheKey(et, mat, "normal", "default")
            ms = []
            for v in range(a.seeds):
                torch.cuda.synchronize()
                t = time.perf_counter()
                raw = g(build(et, mat, "normal", "default"), seed_for(k, v))
                torch.cuda.synchronize()
                ms.append((time.perf_counter() - t) * 1000.0)
                y = ws.finish(raw, et, "normal", True, True)
                d = OUT / ("%s_%d" % (a.model, fr))
                d.mkdir(parents=True, exist_ok=True)
                AudioClip(y, ws.SR).to_wav(str(d / ("%s-%s_%02d.wav" % (et.replace(" ", "_"), mat, v))))
            rep["rows"].append({"frames": fr, "key": "%s|%s" % (et, mat), "median_ms": float(np.median(ms))})
            print("%s frames %d %-22s %.0f ms" % (a.model, fr, et + "|" + mat, np.median(ms)), flush=True)
    rep["peak_gb"] = torch.cuda.max_memory_reserved() / 1e9
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(rep, open(OUT / ("%s.json" % a.model), "w"), indent=1)
    return 0


def gen_for(kind, frames, cache):
    if "model" not in cache:
        cache["model"] = ws.load(kind)
    torch, ldm, cfg = cache["model"]
    if kind == "flow":
        from woosh.inference.flowmatching_sampler import flowmatching_integrate
    else:
        from woosh.inference.flowmap_sampler import sample_euler

    def gen(prompt, seed):
        torch.manual_seed(int(seed))
        noise = torch.randn(1, 128, int(frames)).to("cuda")
        cond = ldm.get_cond({"audio": None, "description": [prompt]}, no_dropout=True, device="cuda")
        with torch.inference_mode():
            if kind == "flow":
                x = flowmatching_integrate(ldm, noise=noise, cond=cond, cfg=cfg, atol=0.001, rtol=0.001,
                                           device="cuda", dtype=torch.float64)
            else:
                x = sample_euler(model=ldm, noise=noise, cond=cond, num_steps=4, renoise=[0, 0.5, 0.5, 0.3], cfg=cfg)
            audio = ldm.autoencoder.inverse(x)
        return audio[0].detach().to(torch.float32).cpu().numpy().T

    return gen


if __name__ == "__main__":
    raise SystemExit(main())

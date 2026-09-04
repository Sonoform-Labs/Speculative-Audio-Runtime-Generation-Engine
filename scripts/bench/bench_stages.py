import _bootstrap

import time

import numpy as np
import torch

from sarge.generators.stable_audio3 import StableAudio3Generator
from sarge.generators.stable_audio import _extract, build_prompt
from sarge.loudness import normalize_loudness, target_for
from sarge.types import CacheKey

KEYS = [
    CacheKey("footstep", "wood", "normal"),
    CacheKey("impact", "glass", "normal"),
    CacheKey("hit", "metal", "normal"),
    CacheKey("scrape", "metal", "normal"),
    CacheKey("slam", "door", "normal"),
]
N = 8


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class Timer:
    def __init__(self):
        self.t = {}

    def __call__(self, name, fn):
        sync()
        t0 = time.perf_counter()
        out = fn()
        sync()
        self.t[name] = self.t.get(name, 0.0) + (time.perf_counter() - t0) * 1000.0
        return out


def find_conditioner(m):
    seen = set()

    def walk(o, d=0):
        if d > 5 or id(o) in seen:
            return None
        seen.add(id(o))
        for n, v in getattr(o, "__dict__", {}).items():
            if "condition" in n.lower() and hasattr(v, "__call__"):
                return v
            r = walk(v, d + 1)
            if r is not None:
                return r
        return None

    return walk(m)


def main():
    g = StableAudio3Generator()
    g.ensure_ready()
    m = g._model
    dev = g.device

    for k in KEYS[:2]:
        g.generate(k, variant=0)

    print(f"device={dev} steps={g.steps} sampler={g.sampler_type} cfg={g.cfg_scale}")
    print(f"gen_dur={g.gen_duration_s}s out_dur={g.duration_s}s sr={g.sample_rate}\n")

    tm = Timer()
    total = 0.0
    for i in range(N):
        key = KEYS[i % len(KEYS)]
        sync()
        t_all = time.perf_counter()

        prompt = tm("1_build_prompt", lambda: build_prompt(key.event_type, key.material, key.intensity))

        audio = tm("2_model_generate", lambda: m.generate(
            prompt=prompt, duration=g.gen_duration_s, steps=g._steps_for(key),
            cfg_scale=g.cfg_scale, seed=1234 + i, sampler_type=g.sampler_type,
            disable_tqdm=True))

        a = tm("3_to_numpy", lambda: audio.detach().to(torch.float32).cpu().numpy())
        arr = a
        while arr.ndim > 2:
            arr = arr[0]
        mono = arr.mean(axis=0) if arr.ndim == 2 else arr
        seg = tm("4_extract", lambda: _extract(mono, g.sample_rate, g.duration_s))
        tm("5_loudness", lambda: normalize_loudness(seg, target_rms=target_for(key.intensity)))
        tm("6_pcm_encode", lambda: seg.astype(np.float32).tobytes())

        sync()
        total += (time.perf_counter() - t_all) * 1000.0

    print(f"{'stage':20}{'ms/gen':>10}{'% total':>10}")
    print("-" * 40)
    tot = total / N
    for k in sorted(tm.t):
        v = tm.t[k] / N
        print(f"{k:20}{v:10.1f}{100*v/tot:9.1f}%")
    print("-" * 40)
    print(f"{'TOTAL':20}{tot:10.1f}")

    print("\ncfg sweep (model.generate only, same seed/steps):")
    for cfg in (2.0, 1.5, 1.0):
        ts = []
        for i in range(5):
            sync()
            t0 = time.perf_counter()
            m.generate(prompt=build_prompt("footstep", "wood", "normal"),
                       duration=g.gen_duration_s, steps=g.steps, cfg_scale=cfg,
                       seed=99, sampler_type=g.sampler_type, disable_tqdm=True)
            sync()
            ts.append((time.perf_counter() - t0) * 1000.0)
        print(f"  cfg={cfg:<5} median {sorted(ts)[len(ts)//2]:6.1f} ms")

    print("\nstep sweep (cfg=1.0):")
    for st in (8, 4, 3, 2, 1):
        ts = []
        for i in range(5):
            sync()
            t0 = time.perf_counter()
            m.generate(prompt=build_prompt("footstep", "wood", "normal"),
                       duration=g.gen_duration_s, steps=st, cfg_scale=1.0,
                       seed=99, sampler_type=g.sampler_type, disable_tqdm=True)
            sync()
            ts.append((time.perf_counter() - t0) * 1000.0)
        med = sorted(ts)[len(ts)//2]
        print(f"  steps={st:<3} median {med:6.1f} ms")

    print("\nfixed-overhead estimate: intercept of the step sweep = per-gen cost independent of steps")
    print("if intercept is large, the text encoder / VAE decode dominate and caching wins")


if __name__ == "__main__":
    main()

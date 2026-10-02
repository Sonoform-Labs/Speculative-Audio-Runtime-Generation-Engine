import _bootstrap

import json
import os
import time

import numpy as np
import torch

from sarge.generators.sa3_fast import Accelerator
from sarge.generators.sa3_hoist import install_hoist
from sarge.generators.stable_audio3 import StableAudio3Generator
from sarge.types import CacheKey

KEYS = [CacheKey("footstep", "wood", "normal"), CacheKey("footstep", "metal", "heavy"),
        CacheKey("impact", "glass", "normal"), CacheKey("footstep", "water", "normal")]
PAIRS = [[(CacheKey("footstep", "stone", "normal"), 3), (CacheKey("footstep", "grass", "normal"), 4)],
         [(CacheKey("impact", "metal", "normal"), 1), (CacheKey("footstep", "wood", "light"), 2)]]


def timed(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, (time.perf_counter() - t0) * 1000.0


def run(g, reps):
    single, ms = {}, {}
    for k in KEYS:
        g.generate(k, variant=0)
        ts = []
        for _ in range(reps):
            c, t = timed(lambda: g.generate(k, variant=0))
            ts.append(t)
        single[str(k)] = c.samples
        ms[str(k)] = float(np.median(ts))
    batch, bms = {}, []
    for i, pair in enumerate(PAIRS):
        g.generate_batch(pair)
        ts = []
        for _ in range(reps):
            cs, t = timed(lambda: g.generate_batch(pair))
            ts.append(t)
        batch[i] = [c.samples for c in cs]
        bms.append(float(np.median(ts)))
    return single, ms, batch, bms


def main():
    reps = 6
    g = StableAudio3Generator(fast=False)
    g.ensure_ready()
    ref, ref_ms, ref_b, ref_bms = run(g, reps)
    g._fast = Accelerator(g._model)
    g._hoist = install_hoist(g._model, g._fast, cache_prompts=8)
    fast, fast_ms, fast_b, fast_bms = run(g, reps)
    again, _, _, _ = run(g, 1)

    res = {"single": {}, "batch": [], "stats": g.fast_stats()}
    print()
    print("%-34s %9s %9s %9s %12s %6s" % ("key", "ref ms", "fast ms", "saved", "max|diff|", "exact"))
    for k in KEYS:
        s = str(k)
        d = float(np.max(np.abs(ref[s] - fast[s])))
        ex = bool(np.array_equal(ref[s], fast[s]))
        rep = bool(np.array_equal(fast[s], again[s]))
        res["single"][s] = {"ref_ms": ref_ms[s], "fast_ms": fast_ms[s], "max_abs_diff": d, "exact": ex,
                            "repeatable": rep}
        print("%-34s %9.1f %9.1f %9.1f %12.2e %6s" % (s, ref_ms[s], fast_ms[s], ref_ms[s] - fast_ms[s], d, ex))
    for i, pair in enumerate(PAIRS):
        ds = [float(np.max(np.abs(a - b))) for a, b in zip(ref_b[i], fast_b[i])]
        ex = all(np.array_equal(a, b) for a, b in zip(ref_b[i], fast_b[i]))
        res["batch"].append({"pair": [str(k) for k, _ in pair], "ref_ms": ref_bms[i], "fast_ms": fast_bms[i],
                             "max_abs_diff": ds, "exact": ex})
        print("%-34s %9.1f %9.1f %9.1f %12.2e %6s" % ("batch2 " + str(pair[0][0].material) + "+" + str(pair[1][0].material),
                                                     ref_bms[i], fast_bms[i], ref_bms[i] - fast_bms[i], max(ds), ex))
    print("accelerator", res["stats"])
    json.dump(res, open(os.path.join("scripts", "bench", "out_fast_equiv.json"), "w"), indent=1)
    print("saved -> scripts\\bench\\out_fast_equiv.json")


if __name__ == "__main__":
    main()

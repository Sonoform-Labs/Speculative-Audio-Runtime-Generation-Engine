import os
import sys

BENCH = os.path.join(os.getcwd(), "scripts", "bench")
if not os.path.isfile(os.path.join(BENCH, "_bootstrap.py")):
    BENCH = r"D:\Python\SAGE\scripts\bench"
sys.path.insert(0, BENCH)

import _bootstrap

import json
import time
from collections import defaultdict

import numpy as np
import torch

from sarge.generators import stable_audio3 as sa3
from sarge.generators.stable_audio3 import StableAudio3Generator
from sarge.types import CacheKey

ROOT = os.path.dirname(os.path.dirname(BENCH))
KEYS = [("footstep", "wood"), ("shoot", "rifle"), ("break", "window"), ("slam", "door")]
CFG = {"sampler_type": "euler", "steps": 3, "detail_steps": 3, "compile_mode": "default", "compile_decode": True,
       "graph_decode": True, "hoist_rope": True, "skip_first_step": True}


def main():
    g = StableAudio3Generator(**CFG)
    g.ensure_ready()
    times = defaultdict(list)
    sam = g._model
    pt = sam.model.pretransform
    orig_gen, orig_finish, orig_guard, orig_dec = sam.generate, g._finish, g._guard, pt.decode

    def timed(name, fn, sync=True):
        def w(*a, **k):
            if sync:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = fn(*a, **k)
            if sync:
                torch.cuda.synchronize()
            times[name].append((time.perf_counter() - t0) * 1000.0)
            return out
        return w

    sam.generate = timed("model.generate (sampling + decode)", orig_gen)
    pt.decode = timed("decode", orig_dec)
    g._finish = timed("_finish (to cpu, extract, loudness)", orig_finish, sync=False)
    g._guard = timed("static guard", orig_guard, sync=False)
    for et, mat in KEYS:
        g.generate(CacheKey(et, mat, "normal"), variant=0)
    times.clear()
    total = []
    for rep in range(5):
        for et, mat in KEYS:
            for v in range(4):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                g.generate(CacheKey(et, mat, "normal"), variant=v)
                torch.cuda.synchronize()
                total.append((time.perf_counter() - t0) * 1000.0)
    res = {"total_median_ms": float(np.median(total))}
    print("total generate(): median %.2f ms" % res["total_median_ms"])
    for k, v in times.items():
        res[k] = float(np.median(v))
        print("  %-38s median %.2f ms" % (k, res[k]))
    json.dump(res, open(os.path.join(ROOT, "scripts", "bench", "out_breakdown.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

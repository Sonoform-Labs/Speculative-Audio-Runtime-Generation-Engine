import os
import sys

BENCH = os.path.join(os.getcwd(), "scripts", "bench")
if not os.path.isfile(os.path.join(BENCH, "_bootstrap.py")):
    BENCH = r"D:\Python\SAGE\scripts\bench"
sys.path.insert(0, BENCH)

import _bootstrap

import json
import time

import numpy as np
import torch

import vocab_eval as ve
from sarge.generators.stable_audio3 import StableAudio3Generator
from sarge.types import CacheKey

ROOT = os.path.dirname(os.path.dirname(BENCH))
CFG = {"sampler_type": "euler", "steps": 3, "compile_mode": "default", "compile_decode": True,
       "graph_decode": True, "hoist_rope": True, "skip_first_step": True, "fp16_accumulation": True, "fast_load": True, "unfuse_bias": "--tg" in sys.argv, "triton_gemm": "--tg" in sys.argv}


def timed(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, (time.perf_counter() - t0) * 1000.0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    cfg = dict(CFG, graph_text="--graph-text" in argv)
    g = StableAudio3Generator(**cfg)
    g.ensure_ready()
    keys = [CacheKey(et, mat, ve.INTENSITY) for et, mat, _t in ve.KEYS]
    _, warm = timed(lambda: g.generate(keys[0], variant=0))
    rows = []
    for k in keys[1:]:
        before = g.fast_stats()
        _, miss = timed(lambda: g.generate(k, variant=0))
        mid = g.fast_stats()
        _, hit = timed(lambda: g.generate(k, variant=1))
        rows.append({"key": str(k), "miss_ms": miss, "hit_ms": hit,
                     "cond_miss": mid.get("cond_misses", 0) - before.get("cond_misses", 0),
                     "moved": {n: mid[n] - before.get(n, 0) for n in mid
                               if type(mid[n]) is int and mid[n] != before.get(n, 0)}})
        print("%-24s miss %6.1f ms  repeat %6.1f ms  %s" % (k, miss, hit, rows[-1]["moved"]), flush=True)
    cache = g._fast.cache
    enc = []
    for i in range(10):
        _, ms = timed(lambda: cache._one("a probe prompt number %d, dry, isolated sound effect" % i, g.gen_duration_s))
        enc.append(ms)
    rep = {"warm_ms": warm, "miss_median_ms": float(np.median([r["miss_ms"] for r in rows])),
           "hit_median_ms": float(np.median([r["hit_ms"] for r in rows])),
           "encode_median_ms": float(np.median(enc)), "rows": rows, "config": cfg,
           "torch": torch.__version__}
    print("first use of a prompt %.1f ms, repeat %.1f ms, conditioner alone %.1f ms" % (
        rep["miss_median_ms"], rep["hit_median_ms"], rep["encode_median_ms"]), flush=True)
    if cfg["graph_text"]:
        from sarge.generators.sa3_fast import GraphedText
        conds = g._model.model.conditioner.conditioners
        a = [g.generate(k, variant=7).samples for k in keys[:6]]
        for c in conds.values():
            if isinstance(vars(c).get("model"), GraphedText):
                c.__dict__["model"] = vars(c)["model"].orig
        cache.store.clear()
        b = [g.generate(k, variant=7).samples for k in keys[:6]]
        rep["identical_clips"] = sum(bool(np.array_equal(x, y)) for x, y in zip(a, b))
        rep["max_abs_diff"] = float(max(np.abs(x - y).max() for x, y in zip(a, b)))
        rep["stats"] = {k: v for k, v in g.fast_stats().items() if k.startswith("text")}
        print("graph on vs off: %d of 6 clips identical, max diff %.3g, %s" % (
            rep["identical_clips"], rep["max_abs_diff"], rep["stats"]), flush=True)
    name = "out_first_hit%s%s.json" % ("_text" if cfg["graph_text"] else "", "_tg" if cfg["triton_gemm"] else "")
    json.dump(rep, open(os.path.join(BENCH, name), "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

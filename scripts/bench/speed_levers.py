import os
import sys

BENCH = os.path.join(os.getcwd(), "scripts", "bench")
if not os.path.isfile(os.path.join(BENCH, "_bootstrap.py")):
    BENCH = r"D:\Python\SAGE\scripts\bench"
sys.path.insert(0, BENCH)

import _bootstrap

import argparse
import gc
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from sarge.generators.stable_audio import _seed
from sarge.generators.stable_audio3 import StableAudio3Generator
from sarge.types import CacheKey

ROOT = os.path.dirname(os.path.dirname(BENCH))
KEYS = [("footstep", "wood"), ("shoot", "rifle"), ("break", "window"), ("slam", "door")]
ARMS = {
    "base_cfg2_s3": {"cfg_scale": 2.0, "steps": 3},
    "cfg1_s3": {"cfg_scale": 1.0, "steps": 3},
    "cfg2_first2_s3": {"cfg_scale": 2.0, "steps": 3, "cfg_interval": (0.9, 1.0)},
    "cfg2_first1_s3": {"cfg_scale": 2.0, "steps": 3, "cfg_interval": (0.99, 1.0)},
    "cfg2_s2": {"cfg_scale": 2.0, "steps": 2},
    "cfg1_s2": {"cfg_scale": 1.0, "steps": 2},
}
GROUPS = [("gemm", ("mm", "gemm", "linear", "matmul", "cutlass", "sm80_xmma", "ampere_")),
          ("attention", ("scaled_dot_product", "attention", "flash", "fmha")),
          ("conv", ("conv", "cudnn", "winograd", "implicit")),
          ("norm", ("norm",)),
          ("elementwise", ("elementwise", "vectorized", "unrolled", "mul", "add", "copy", "cat", "silu", "gelu",
                           "sigmoid", "tanh", "exp", "sin", "cos", "pow", "where", "fill"))]


def parse(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--profile-reps", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(ROOT, "scripts", "bench", "out_speed_levers.json"))
    return ap.parse_args(argv)


def sync():
    torch.cuda.synchronize()


def gen(g, key, variant, arm):
    kw = dict(arm)
    steps = kw.pop("steps")
    cfg = kw.pop("cfg_scale")
    with torch.no_grad():
        return g._model.generate(duration=g.gen_duration_s, steps=steps, cfg_scale=cfg,
                                 seed=_seed(key, variant), **g._inputs([g._prompt(key)]), **kw)


def time_arms(g, args):
    out = {}
    for name, arm in ARMS.items():
        try:
            for et, mat in KEYS[:2]:
                gen(g, CacheKey(et, mat, "normal"), 0, arm)
        except Exception as e:
            out[name] = {"error": repr(e)[:300]}
            print("%-16s failed: %s" % (name, out[name]["error"]), flush=True)
            continue
        ms = []
        for et, mat in KEYS:
            k = CacheKey(et, mat, "normal")
            for v in range(args.reps):
                sync()
                t0 = time.perf_counter()
                gen(g, k, v, arm)
                sync()
                ms.append((time.perf_counter() - t0) * 1000.0)
        out[name] = {"median_ms": float(np.median(ms)), "p10_ms": float(np.percentile(ms, 10)),
                     "p90_ms": float(np.percentile(ms, 90)), "n": len(ms)}
        print("%-16s median %6.1f ms  p10 %6.1f  p90 %6.1f" % (name, out[name]["median_ms"], out[name]["p10_ms"],
                                                               out[name]["p90_ms"]), flush=True)
    return out


def decode_ms(g, reps=20):
    pt = g._model.model.pretransform
    dtype = next(pt.parameters()).dtype
    z = torch.randn(1, 256, 22, device="cuda", dtype=dtype)
    with torch.no_grad():
        for _ in range(3):
            pt.decode(z)
        ms = []
        for _ in range(reps):
            sync()
            t0 = time.perf_counter()
            pt.decode(z)
            sync()
            ms.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(ms))


def sdpa_backends():
    from torch.nn.attention import SDPBackend, sdpa_kernel
    q = torch.randn(2, 16, 86, 64, device="cuda", dtype=torch.float16)
    k = torch.randn(2, 16, 257, 64, device="cuda", dtype=torch.float16)
    res = {}
    for name, be in (("flash", SDPBackend.FLASH_ATTENTION), ("efficient", SDPBackend.EFFICIENT_ATTENTION),
                     ("cudnn", SDPBackend.CUDNN_ATTENTION), ("math", SDPBackend.MATH)):
        try:
            with sdpa_kernel(be):
                F.scaled_dot_product_attention(q, q, q)
                F.scaled_dot_product_attention(q, k, k)
            res[name] = "ok"
        except Exception as e:
            res[name] = repr(e)[:120]
    return res


def device_time(ev):
    for a in ("self_device_time_total", "self_cuda_time_total"):
        v = getattr(ev, a, None)
        if v is not None:
            return float(v)
    return 0.0


def profile_eager(args):
    from torch.profiler import ProfilerActivity, profile
    g = StableAudio3Generator(sampler_type="euler", steps=3, detail_steps=3, fast=False, static_guard=False)
    g.ensure_ready()
    k = CacheKey("footstep", "wood", "normal")
    arm = ARMS["base_cfg2_s3"]
    for v in range(2):
        gen(g, k, v, arm)
    sync()
    t0 = time.perf_counter()
    for v in range(args.profile_reps):
        gen(g, k, v, arm)
    sync()
    eager_ms = (time.perf_counter() - t0) * 1000.0 / args.profile_reps
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for v in range(args.profile_reps):
            gen(g, k, v, arm)
        sync()
    rows = []
    for ev in prof.key_averages():
        d = device_time(ev)
        if d > 0:
            rows.append((ev.key, d / 1000.0 / args.profile_reps, int(ev.count / args.profile_reps)))
    rows.sort(key=lambda r: -r[1])
    total = sum(r[1] for r in rows)
    groups = {name: 0.0 for name, _ in GROUPS}
    groups["other"] = 0.0
    for name, ms, _n in rows:
        low = name.lower()
        for gname, pats in GROUPS:
            if any(p in low for p in pats):
                groups[gname] += ms
                break
        else:
            groups["other"] += ms
    del g
    gc.collect()
    torch.cuda.empty_cache()
    return {"eager_wall_ms": eager_ms, "kernel_ms_per_clip": total, "groups_ms": groups,
            "top": [{"name": n[:110], "ms": m, "calls": c} for n, m, c in rows[:25]]}


def main(argv=None):
    args = parse(argv)
    torch.backends.cuda.matmul.allow_tf32 = True
    res = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__, "keys": KEYS}
    res["sdpa_backends"] = sdpa_backends()
    print("sdpa backends:", res["sdpa_backends"], flush=True)
    g = StableAudio3Generator(sampler_type="euler", steps=3, detail_steps=3, static_guard=False)
    g.ensure_ready()
    res["decode_ms"] = decode_ms(g)
    print("decode alone: %.1f ms" % res["decode_ms"], flush=True)
    res["arms"] = time_arms(g, args)
    res["fast_stats"] = g.fast_stats() if hasattr(g, "fast_stats") else {}
    del g
    gc.collect()
    torch.cuda.empty_cache()
    res["profile"] = profile_eager(args)
    p = res["profile"]
    print("eager wall %.1f ms per clip, kernel time %.1f ms" % (p["eager_wall_ms"], p["kernel_ms_per_clip"]))
    print("kernel time by group:", {k: round(v, 1) for k, v in p["groups_ms"].items()})
    for r in p["top"][:15]:
        print("  %7.2f ms  x%-4d %s" % (r["ms"], r["calls"], r["name"]))
    json.dump(res, open(args.out, "w", encoding="utf-8"), indent=1)
    print("saved -> %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

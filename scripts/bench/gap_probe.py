import os
import sys

BENCH = os.path.join(os.getcwd(), "scripts", "bench")
if not os.path.isfile(os.path.join(BENCH, "_bootstrap.py")):
    BENCH = r"D:\Python\SAGE\scripts\bench"
sys.path.insert(0, BENCH)

import _bootstrap

import json
from collections import defaultdict

import torch
from torch.profiler import ProfilerActivity, profile

from sarge.generators.stable_audio3 import StableAudio3Generator
from sarge.types import CacheKey

CFG = {"sampler_type": "euler", "steps": 3, "compile_mode": "default", "compile_decode": True, "graph_decode": True,
       "hoist_rope": True, "skip_first_step": True, "fp16_accumulation": True, "graph_text": True, "fast_load": True,
       "unfuse_bias": True, "triton_gemm": True, "decode_prefix": True}


def main():
    g = StableAudio3Generator(**CFG)
    g.ensure_ready()
    key = CacheKey("footstep", "wood", "normal")
    for v in range(4):
        g.generate(key, variant=v)
    torch.cuda.synchronize()
    n = 8
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
        for v in range(n):
            with torch.profiler.record_function("clip"):
                g.generate(key, variant=20 + v)
            torch.cuda.synchronize()
    ev = prof.events()
    clips = sorted([(e.time_range.start, e.time_range.end) for e in ev if e.name == "clip"])
    kern = sorted([(e.time_range.start, e.time_range.end, e.name) for e in ev
                   if e.device_type == torch.autograd.DeviceType.CUDA])
    cpu = [e for e in ev if e.device_type == torch.autograd.DeviceType.CPU and e.name != "clip"]
    rows = []
    gap_ops = defaultdict(float)
    for c0, c1 in clips:
        ks = [k for k in kern if k[0] >= c0 and k[1] <= c1]
        busy = 0.0
        last = None
        gaps = []
        for s, e, name in ks:
            if last is None:
                gaps.append((c0, s))
            elif s > last:
                gaps.append((last, s))
            busy += e - max(s, last or s)
            last = max(last or e, e)
        gaps.append((last, c1))
        big = [(a, b) for a, b in gaps if b - a > 30]
        for a, b in big:
            inside = [x for x in cpu if x.time_range.start >= a - 1 and x.time_range.end <= b + 1]
            top = sorted(inside, key=lambda x: -(x.time_range.end - x.time_range.start))[:1]
            gap_ops[top[0].name if top else "none"] += (b - a) / 1000.0
        rows.append({"wall_ms": (c1 - c0) / 1000.0, "gpu_busy_ms": busy / 1000.0,
                     "first_kernel_ms": (ks[0][0] - c0) / 1000.0 if ks else None,
                     "after_last_kernel_ms": (c1 - last) / 1000.0 if ks else None,
                     "gaps_over_30us_ms": sum(b - a for a, b in big) / 1000.0, "n_gaps": len(big)})
    rep = {"per_clip": rows, "gap_owner_ms_total": dict(sorted(gap_ops.items(), key=lambda kv: -kv[1])[:20])}
    for r in rows:
        print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
    print(json.dumps({k: round(v / n, 3) for k, v in rep["gap_owner_ms_total"].items()}, indent=1), flush=True)
    json.dump(rep, open(os.path.join(BENCH, "out_gap_probe.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
       "hoist_rope": True, "skip_first_step": True, "fp16_accumulation": True, "graph_text": True, "fast_load": True}


def bucket(name):
    n = name.lower()
    if "gemm" in n or "cutlass" in n or "sm80" in n or "sm86" in n or "ampere" in n or "s16816" in n or "s1688" in n:
        return "matmul"
    if "flash" in n or "fmha" in n or "attention" in n or "efficient" in n:
        return "attention"
    if n.startswith("triton"):
        return "triton fused"
    if "conv" in n or "implicit" in n or "winograd" in n:
        return "conv"
    if "memcpy" in n or "memset" in n:
        return "copy"
    return "other"


def main():
    g = StableAudio3Generator(**CFG)
    g.ensure_ready()
    key = CacheKey("footstep", "wood", "normal")
    for v in range(3):
        g.generate(key, variant=v)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
        for v in range(10):
            g.generate(key, variant=10 + v)
        torch.cuda.synchronize()
    per = defaultdict(float)
    count = defaultdict(int)
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            per[e.name] += e.device_time_total
            count[e.name] += 1
    total = sum(per.values())
    groups = defaultdict(float)
    for n, t in per.items():
        groups[bucket(n)] += t
    top = sorted(per.items(), key=lambda kv: -kv[1])[:30]
    rep = {"gpu_ms_per_clip": total / 10 / 1000.0,
           "groups_ms_per_clip": {k: v / 10 / 1000.0 for k, v in sorted(groups.items(), key=lambda kv: -kv[1])},
           "top": [{"name": n[:160], "ms_per_clip": t / 10 / 1000.0, "calls_per_clip": count[n] / 10} for n, t in top]}
    print(json.dumps(rep, indent=1), flush=True)
    json.dump(rep, open(os.path.join(BENCH, "out_kernel_profile.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import _bootstrap

import json
import os
import time

import numpy as np
import torch
from torch.profiler import profile, ProfilerActivity

from sarge.generators.stable_audio3 import StableAudio3Generator
from sarge.generators.stable_audio import build_prompt


def sync():
    torch.cuda.synchronize()


def main():
    g = StableAudio3Generator()
    g.ensure_ready()
    m = g._model
    p = build_prompt("footstep", "wood", "normal")
    kw = dict(duration=g.gen_duration_s, steps=3, cfg_scale=1.0,
              sampler_type=g.sampler_type, disable_tqdm=True)
    for _ in range(3):
        m.generate(prompt=p, seed=1, **kw)

    res = {"batch": {}}
    print("batch scaling (steps=3):")
    for b in (1, 2, 4, 8):
        ts = []
        for i in range(5):
            sync()
            t0 = time.perf_counter()
            m.generate(prompt=[p] * b, batch_size=b, seed=7 + i, **kw)
            sync()
            ts.append((time.perf_counter() - t0) * 1000.0)
        med = float(np.median(ts))
        res["batch"][b] = {"ms": med, "ms_per_clip": med / b}
        print("  batch %d   %7.1f ms   %6.1f ms/clip" % (b, med, med / b))

    sync()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        t0 = time.perf_counter()
        m.generate(prompt=p, seed=3, **kw)
        sync()
        wall = (time.perf_counter() - t0) * 1000.0
    ev = prof.key_averages()
    cuda_us = sum(e.self_device_time_total for e in ev)
    launches = sum(e.count for e in ev if "cudaLaunchKernel" in e.key)
    res["profile"] = {"wall_ms": wall, "gpu_kernel_ms": cuda_us / 1000.0,
                      "gpu_busy_frac": (cuda_us / 1000.0) / wall,
                      "kernel_launches": int(launches)}
    print()
    print("one generate, steps=3:")
    print("  wall            %7.1f ms" % wall)
    print("  gpu kernel time %7.1f ms   (%.0f%% busy)" % (cuda_us / 1000.0,
                                                           100 * cuda_us / 1000.0 / wall))
    print("  kernel launches %7d" % launches)
    print()
    print(prof.key_averages().table(sort_by="self_device_time_total", row_limit=12))
    json.dump(res, open(os.path.join("scripts", "bench", "out_bench_launch.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

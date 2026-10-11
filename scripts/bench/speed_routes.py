import os
import sys

BENCH = os.path.join(os.getcwd(), "scripts", "bench")
if not os.path.isfile(os.path.join(BENCH, "_bootstrap.py")):
    BENCH = r"D:\Python\SAGE\scripts\bench"
sys.path.insert(0, BENCH)

import _bootstrap

import argparse
import json
import time

import numpy as np

import vocab_eval as ve
from sarge.generators.stable_audio3 import StableAudio3Generator
from sarge.types import CacheKey

ROOT = os.path.dirname(os.path.dirname(BENCH))
OUT = os.path.join("data", "_gate", "speed_routes")
ARMS = {
    "comp": {"compile_mode": "default", "compile_decode": True},
    "comp_cfgfirst": {"compile_mode": "default", "compile_decode": True, "cfg_interval": (0.99, 1.0)},
    "comp_cfg1": {"compile_mode": "default", "compile_decode": True, "cfg_scale": 1.0},
    "comp_s2": {"compile_mode": "default", "compile_decode": True, "steps": 2, "detail_steps": 2},
    "comp_cfg1_s2": {"compile_mode": "default", "compile_decode": True, "cfg_scale": 1.0, "steps": 2,
                     "detail_steps": 2},
    "full": {"compile_mode": "default", "compile_decode": True, "graph_decode": True, "hoist_rope": True},
    "full_skip1": {"compile_mode": "default", "compile_decode": True, "graph_decode": True, "hoist_rope": True,
                   "skip_first_step": True},
}


def parse(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--out-dir", default=OUT)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse(argv)
    import torch
    base = ve.SETTINGS[ve.CURRENT]
    cfg = {"sampler_type": "euler", "steps": base["steps"], "detail_steps": base["detail_steps"],
           "gen_duration_s": base["gen_duration_s"], "duration_s": base["duration_s"]}
    cfg.update(ARMS[args.arm])
    g = StableAudio3Generator(**cfg)
    g.ensure_ready()
    d = os.path.join(ROOT, args.out_dir, args.arm)
    os.makedirs(d, exist_ok=True)
    keys = [(et, mat) for et, mat, _t in ve.KEYS]
    t0 = time.perf_counter()
    g.generate(CacheKey(keys[0][0], keys[0][1], ve.INTENSITY), variant=0)
    warm_s = time.perf_counter() - t0
    prompts, ms = {}, []
    for et, mat in keys:
        k = CacheKey(et, mat, ve.INTENSITY)
        prompts["%s|%s" % (et, mat)] = g._prompt(k)
        for v in range(args.seeds):
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            c = g.generate(k, variant=v)
            torch.cuda.synchronize()
            ms.append((time.perf_counter() - t1) * 1000.0)
            c.to_wav(os.path.join(d, "%s-%s_%02d.wav" % (et, mat, v)))
    info = {"steps": cfg["steps"], "prompts": prompts, "clusters": [], "sources": [], "audio": "none",
            "note": "base small-sfx with speed arm %s: %s" % (args.arm, json.dumps(ARMS[args.arm])),
            "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
            "median_ms": float(np.median(ms)), "warm_s": warm_s, "static": dict(g.static_stats),
            "fast_stats": g.fast_stats() if hasattr(g, "fast_stats") else {}}
    json.dump(info, open(os.path.join(d, "route.json"), "w", encoding="utf-8"), indent=1)
    print("%s  median %.1f ms per clip (first-key warm-up %.1f s)  static %s" % (
        args.arm, info["median_ms"], warm_s, info["static"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

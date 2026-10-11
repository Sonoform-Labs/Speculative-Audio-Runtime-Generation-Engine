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

ROOT = os.path.dirname(os.path.dirname(BENCH))
OUT = os.path.join("data", "_gate", "compile_probe")
KEYS = [("footstep", "wood"), ("footstep", "grass"), ("shoot", "rifle"), ("shoot", "pistol"),
        ("break", "window"), ("slam", "door"), ("bark", "dog"), ("splash", "water")]
ARMS = {
    "base": {},
    "c_default": {"compile_mode": "default"},
    "c_default_dec": {"compile_mode": "default", "compile_decode": True},
    "c_maxat": {"compile_mode": "max-autotune-no-cudagraphs"},
    "c_maxat_dec": {"compile_mode": "max-autotune-no-cudagraphs", "compile_decode": True},
    "gdec": {"graph_decode": True},
    "rope": {"hoist_rope": True},
    "skip1": {"skip_first_step": True},
    "comp_full": {"compile_mode": "default", "compile_decode": True, "graph_decode": True, "hoist_rope": True},
    "comp_full_skip1": {"compile_mode": "default", "compile_decode": True, "graph_decode": True, "hoist_rope": True,
                        "skip_first_step": True},
    "deflt26": {"compile_mode": "default", "compile_decode": True, "graph_decode": True, "hoist_rope": True,
                "skip_first_step": True},
    "deflt214": {"compile_mode": "default", "compile_decode": True, "graph_decode": True, "hoist_rope": True,
                 "skip_first_step": True},
    "deflt214_acc": {"compile_mode": "default", "compile_decode": True, "graph_decode": True, "hoist_rope": True,
                     "skip_first_step": True, "fp16_accumulation": True},
    "deflt214_unf": {"compile_mode": "default", "compile_decode": True, "graph_decode": True, "hoist_rope": True,
                     "skip_first_step": True, "fp16_accumulation": True, "unfuse_bias": True},
    "deflt214_tg": {"compile_mode": "default", "compile_decode": True, "graph_decode": True, "hoist_rope": True,
                    "skip_first_step": True, "fp16_accumulation": True, "unfuse_bias": True, "triton_gemm": True},
}
COMPARE = ["c_default", "c_default_dec", "c_maxat", "c_maxat_dec", "gdec", "rope", "skip1", "comp_full",
           "comp_full_skip1", "deflt26", "deflt214", "deflt214_acc", "deflt214_unf", "deflt214_tg"]


def parse(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARMS) + ["compare"])
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out-dir", default=OUT)
    return ap.parse_args(argv)


def run_arm(args):
    import torch
    from sarge.generators.stable_audio3 import StableAudio3Generator
    from sarge.types import CacheKey
    cfg = ARMS[args.arm]
    g = StableAudio3Generator(sampler_type="euler", steps=3, detail_steps=3, static_guard=False, **cfg)
    t0 = time.perf_counter()
    g.ensure_ready()
    load_s = time.perf_counter() - t0
    d = os.path.join(ROOT, args.out_dir, args.arm)
    os.makedirs(d, exist_ok=True)
    first = {}
    for et, mat in KEYS:
        k = CacheKey(et, mat, "normal")
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        g.generate(k, variant=0)
        torch.cuda.synchronize()
        first["%s|%s" % (et, mat)] = time.perf_counter() - t1
    ms = []
    for rep in range(args.reps):
        for et, mat in KEYS:
            k = CacheKey(et, mat, "normal")
            for v in range(args.seeds):
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                c = g.generate(k, variant=v)
                torch.cuda.synchronize()
                ms.append((time.perf_counter() - t1) * 1000.0)
                if rep == 0:
                    c.to_wav(os.path.join(d, "%s-%s_%02d.wav" % (et, mat, v)))
    rec = {}
    try:
        from torch._dynamo.utils import counters
        rec = {k: dict(v) for k, v in counters.items() if k in ("stats", "recompiles", "graph_break")}
    except Exception as e:
        rec = {"error": repr(e)[:200]}
    res = {"arm": args.arm, "config": cfg, "load_s": load_s, "first_call_s": first,
           "first_call_total_s": sum(first.values()), "median_ms": float(np.median(ms)),
           "p10_ms": float(np.percentile(ms, 10)), "p90_ms": float(np.percentile(ms, 90)), "n": len(ms),
           "fast_stats": g.fast_stats() if hasattr(g, "fast_stats") else {}, "dynamo": rec,
           "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9}
    json.dump(res, open(os.path.join(d, "result.json"), "w", encoding="utf-8"), indent=1)
    print("%s  median %.1f ms  p10 %.1f  p90 %.1f  first calls %.1f s  load %.1f s  vram %.2f GB  dynamo %s" % (
        args.arm, res["median_ms"], res["p10_ms"], res["p90_ms"], res["first_call_total_s"], load_s,
        res["peak_vram_gb"], json.dumps(rec)[:300]), flush=True)
    return 0


def logmel(x, sr=44100):
    from scipy.signal import stft
    _f, _t, z = stft(x.astype(np.float64), fs=sr, nperseg=2048, noverlap=1536)
    p = np.abs(z) ** 2
    nb = 64
    edges = np.geomspace(40, sr / 2, nb + 1)
    f = np.linspace(0, sr / 2, p.shape[0])
    bands = np.stack([p[(f >= edges[i]) & (f < edges[i + 1])].sum(0) for i in range(nb)])
    return 10.0 * np.log10(bands + 1e-10)


def compare(args):
    import vocab_eval as ve
    base = os.path.join(ROOT, args.out_dir, "base")
    out = {}
    seed_d = []
    for et, mat in KEYS:
        ms = [logmel(ve.wav(os.path.join(base, "%s-%s_%02d.wav" % (et, mat, v)))) for v in range(args.seeds)]
        for i in range(len(ms) - 1):
            seed_d.append(float(np.mean(np.abs(ms[i] - ms[i + 1]))))
    seed_med = float(np.median(seed_d))
    for arm in COMPARE:
        d = os.path.join(ROOT, args.out_dir, arm)
        if not os.path.isfile(os.path.join(d, "result.json")):
            continue
        dist, mx = [], []
        for et, mat in KEYS:
            for v in range(args.seeds):
                a = ve.wav(os.path.join(base, "%s-%s_%02d.wav" % (et, mat, v)))
                b = ve.wav(os.path.join(d, "%s-%s_%02d.wav" % (et, mat, v)))
                n = min(len(a), len(b))
                mx.append(float(np.max(np.abs(a[:n] - b[:n]))))
                dist.append(float(np.mean(np.abs(logmel(a) - logmel(b)))))
        r = json.load(open(os.path.join(d, "result.json"), encoding="utf-8"))
        out[arm] = {"median_ms": r["median_ms"], "first_call_total_s": r["first_call_total_s"],
                    "max_abs_median": float(np.median(mx)), "max_abs_max": float(np.max(mx)),
                    "logmel_db_median": float(np.median(dist)), "ratio_to_seed": float(np.median(dist)) / seed_med}
        print("%-16s %6.1f ms  first %5.1f s  max|diff| med %.4f max %.4f  log-mel %.3f dB  ratio to a new seed %.3f" % (
            arm, r["median_ms"], r["first_call_total_s"], out[arm]["max_abs_median"], out[arm]["max_abs_max"],
            out[arm]["logmel_db_median"], out[arm]["ratio_to_seed"]))
    b = json.load(open(os.path.join(base, "result.json"), encoding="utf-8"))
    print("base             %6.1f ms  seed-to-seed log-mel %.3f dB" % (b["median_ms"], seed_med))
    json.dump({"base_ms": b["median_ms"], "seed_logmel_db": seed_med, "arms": out},
              open(os.path.join(ROOT, "scripts", "bench", "out_%s.json" % os.path.basename(os.path.normpath(args.out_dir))), "w", encoding="utf-8"), indent=1)
    return 0


def main(argv=None):
    args = parse(argv)
    return compare(args) if args.arm == "compare" else run_arm(args)


if __name__ == "__main__":
    raise SystemExit(main())

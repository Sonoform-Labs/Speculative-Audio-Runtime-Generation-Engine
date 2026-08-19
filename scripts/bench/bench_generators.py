import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sage.generators import get_generator
from sage.types import CacheKey

KEYS = [
    CacheKey("footstep", "grass", "normal"),
    CacheKey("footstep", "metal", "normal"),
    CacheKey("shoot", "metal", "normal"),
    CacheKey("impact", "wood", "normal"),
    CacheKey("land", "stone", "hard"),
]


def _pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p))]


def bench(backend, steps_list, runs, cfg):
    gen = get_generator(backend)
    if cfg is not None and hasattr(gen, "cfg_scale"):
        gen.cfg_scale = cfg
    t0 = time.perf_counter()
    ready = getattr(gen, "ensure_ready", None)
    if ready:
        ready()
    gen.generate(KEYS[0], variant=0)
    load_ms = (time.perf_counter() - t0) * 1000.0
    rows = []
    for steps in steps_list:
        if hasattr(gen, "steps"):
            gen.steps = int(steps)
        try:
            gen.generate(KEYS[0], variant=99000)
            times = []
            for i in range(runs):
                k = KEYS[i % len(KEYS)]
                t = time.perf_counter()
                gen.generate(k, variant=10000 + i)
                times.append((time.perf_counter() - t) * 1000.0)
            rows.append((steps, statistics.mean(times), _pct(times, 0.5), _pct(times, 0.95), None))
        except Exception as e:
            rows.append((steps, None, None, None, repr(e)[:140]))
    return load_ms, getattr(gen, "name", backend), rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="sa3")
    ap.add_argument("--steps", default="8,4,2,1")
    ap.add_argument("--runs", type=int, default=12)
    ap.add_argument("--cfg", type=float, default=None)
    args = ap.parse_args()
    steps_list = [int(s) for s in args.steps.split(",")]

    print(f"=== bench backend={args.backend} steps={steps_list} runs={args.runs} cfg={args.cfg} ===")
    try:
        load_ms, name, rows = bench(args.backend, steps_list, args.runs, args.cfg)
    except Exception as e:
        print(f"BACKEND FAILED TO LOAD: {repr(e)[:300]}")
        return
    print(f"loaded '{name}' in {load_ms:.0f} ms (incl first warmup gen)")
    print(f"{'steps':>6}{'mean_ms':>10}{'p50_ms':>9}{'p95_ms':>9}   note")
    for steps, mean, p50, p95, err in rows:
        if mean is None:
            print(f"{steps:>6}{'FAIL':>10}            {err}")
        else:
            print(f"{steps:>6}{mean:>10.1f}{p50:>9.1f}{p95:>9.1f}")


if __name__ == "__main__":
    main()

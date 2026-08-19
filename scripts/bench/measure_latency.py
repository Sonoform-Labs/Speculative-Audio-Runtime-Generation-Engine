import _bootstrap

import argparse
import json
import time

from sage.generators import get_generator
from sage.types import CacheKey


def _vram():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 * 1024), torch.cuda.get_device_name(0)
    except Exception:
        pass
    return None, None


def _reset_vram():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Measure real generation latency for a backend.")
    p.add_argument("--generator", default="stableaudio")
    p.add_argument("--variants", type=int, default=8)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--event-type", default="footstep", dest="event_type")
    p.add_argument("--materials", nargs="+", default=["wood", "stone", "grass", "metal"])
    p.add_argument("--intensities", nargs="+", default=["normal"])
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    gen = get_generator(args.generator)
    keys = [CacheKey(args.event_type, m, i) for m in args.materials for i in args.intensities]

    _reset_vram()
    for w in range(args.warmup):
        gen.generate(keys[0], variant=10_000 + w)

    latencies = []
    durations = []
    for k in keys:
        for v in range(args.variants):
            t0 = time.perf_counter()
            clip = gen.generate(k, variant=v)
            latencies.append((time.perf_counter() - t0) * 1000.0)
            durations.append(clip.duration_s)

    latencies.sort()
    n = len(latencies)
    mean = sum(latencies) / n
    mean_dur_ms = (sum(durations) / len(durations)) * 1000.0
    p50 = latencies[min(n - 1, int(0.50 * n))]
    p95 = latencies[min(n - 1, int(0.95 * n))]
    vram, device = _vram()

    print(f"generator       : {gen.name}")
    if device:
        print(f"device          : {device}")
    print(f"clips measured  : {n}  ({args.variants} variants x {len(keys)} keys, warmup {args.warmup})")
    print(f"clip duration   : {mean_dur_ms:.0f} ms mean")
    print(f"latency (ms)    : mean {mean:.1f} | p50 {p50:.1f} | p95 {p95:.1f} | "
          f"min {latencies[0]:.1f} | max {latencies[-1]:.1f}")
    print(f"real-time factor: {mean / mean_dur_ms:.3f}  (latency / audio duration)")
    if vram is not None:
        print(f"peak VRAM       : {vram:.0f} MB")
    print()
    print(f"feasibility read: a prediction needs >= ~{p95:.0f} ms of lead time for this backend "
          f"to have audio ready in time (p95).")

    if args.out:
        payload = {
            "generator": gen.name, "device": device, "clips": n,
            "clip_duration_ms": mean_dur_ms,
            "latency_ms": {"mean": mean, "p50": p50, "p95": p95,
                           "min": latencies[0], "max": latencies[-1]},
            "rtf": mean / mean_dur_ms, "peak_vram_mb": vram,
        }
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

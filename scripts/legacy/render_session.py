import _bootstrap

import argparse
import os

import numpy as np

from sage.affordance import AffordanceEngine, make_session
from sage.generators import get_generator
from sage.types import AudioClip


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Render the affordance-engine session to one audio timeline.")
    ap.add_argument("--generator", default="sa3")
    ap.add_argument("--out", default=os.path.join("data", "demo", "session_render.wav"))
    ap.add_argument("--variants", type=int, default=3)
    args = ap.parse_args(argv)

    gen = get_generator(args.generator)
    eng = AffordanceEngine(gen, vram_budget_kb=50_000, gpu_ms_budget=10_000_000,
                           gen_cost_ms=getattr(gen, "latency_ms", 393.0), variants_per_key=args.variants)
    session = make_session()
    sr = gen.sample_rate
    placed = []
    for t, kind, payload in session:
        if kind == "context":
            eng.on_context_change(payload)
        else:
            status = eng.on_event(payload)
            placed.append((int(t * sr), eng.last_served.samples.astype(np.float32),
                           status, f"{payload.event_type}/{payload.material}"))

    end = max(off + len(s) for off, s, _, _ in placed)
    buf = np.zeros(end + sr // 2, dtype=np.float32)
    for off, s, _, _ in placed:
        buf[off:off + len(s)] += s
    peak = float(np.max(np.abs(buf))) or 1.0
    buf = (buf / peak * 0.9).astype(np.float32)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    AudioClip(buf, sr).to_wav(args.out)

    s = eng.stats
    total = s["hits"] + s["fallback"]
    print(f"rendered {len(placed)} events over {end / sr:.1f}s -> {args.out}")
    print(f"hit-rate {s['hits'] / total * 100:.0f}%   spec_gen {s['spec_gen']}   demand {s['demand_gen']}   "
          f"wasted {eng.wasted()}   gpu {s['gpu_ms'] / 1000:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

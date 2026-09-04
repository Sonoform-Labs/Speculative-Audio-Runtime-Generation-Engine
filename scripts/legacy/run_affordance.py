import _bootstrap

import argparse
from collections import defaultdict

from sarge.affordance import AffordanceEngine, make_session
from sarge.generators import get_generator


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run the affordance-driven speculative cache engine on a sim session.")
    ap.add_argument("--generator", default="mock")
    ap.add_argument("--gen-cost-ms", type=float, default=393.0, dest="gen_cost_ms")
    ap.add_argument("--vram-kb", type=float, default=800.0, dest="vram_kb")
    ap.add_argument("--gpu-ms-budget", type=float, default=2400.0, dest="gpu_ms_budget")
    args = ap.parse_args(argv)

    gen = get_generator(args.generator)
    eng = AffordanceEngine(gen, vram_budget_kb=args.vram_kb, gpu_ms_budget=args.gpu_ms_budget,
                           gen_cost_ms=args.gen_cost_ms)
    session = make_session()
    per = defaultdict(lambda: [0, 0])
    timeline = []
    for t, kind, payload in session:
        if kind == "context":
            eng.on_context_change(payload)
            timeline.append((t, "ctx", f"equipped={payload.equipped}  surfaces={payload.surfaces}"))
        else:
            res = eng.on_event(payload)
            per[payload.event_type][0 if res == "cached" else 1] += 1
            timeline.append((t, "evt", f"{payload.event_type}/{payload.material} -> {'HIT' if res == 'cached' else 'fallback'}"))

    s = eng.stats
    total = s["hits"] + s["fallback"]
    print(f"\n=== AFFORDANCE ENGINE ({args.generator}, gen_cost {args.gen_cost_ms:.0f}ms, "
          f"vram {args.vram_kb:.0f}KB, budget {args.gpu_ms_budget:.0f}ms/ctx) ===")
    print(f" events {total}   HIT {s['hits']} / fallback {s['fallback']}   = hit-rate {s['hits'] / total * 100:.0f}%")
    print(f" speculative gens {s['spec_gen']}   demand gens {s['demand_gen']}   "
          f"wasted (cached, never fired) {eng.wasted()}   evictions {s['evicted']}")
    print(f" peak cache {s['peak_bytes'] / 1024:.0f} KB   total gpu {s['gpu_ms'] / 1000:.1f} s")
    print(" per event-type (hit/total):")
    for et in sorted(per):
        h, f = per[et]
        print(f"   {et:13s} {h}/{h + f}")
    print("\n timeline:")
    for t, k, msg in timeline:
        print(f"  {t:5.1f}s  {k}  {msg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

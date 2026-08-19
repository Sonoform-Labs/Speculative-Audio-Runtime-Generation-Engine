import _bootstrap

import argparse

import numpy as np

from sage.gamestate import AnticipatoryPredictor, simulate
from sage.generators import get_generator
from sage.metrics import summarize
from sage.pipeline import PredictivePipeline


def match_metrics(preds, events, tol=0.15):
    ev = sorted(events, key=lambda e: e.time_s)
    et = np.array([e.time_s for e in ev]) if ev else np.array([])
    pt = np.array([p.target_time_s for p in preds]) if preds else np.array([])
    timing_hits, correct_hits, leads = 0, 0, []
    for p in preds:
        if et.size == 0:
            break
        j = int(np.argmin(np.abs(et - p.target_time_s)))
        if abs(ev[j].time_s - p.target_time_s) <= tol:
            timing_hits += 1
            if p.key == ev[j].key:
                correct_hits += 1
                leads.append(ev[j].time_s - p.emit_time_s)
    covered = 0
    for e in ev:
        if pt.size == 0:
            break
        near = np.where(np.abs(pt - e.time_s) <= tol)[0]
        if any(preds[k].key == e.key for k in near):
            covered += 1
    precision = correct_hits / len(preds) if preds else 0.0
    recall = covered / len(ev) if ev else 0.0
    key_acc = correct_hits / timing_hits if timing_hits else 0.0
    mean_lead = float(np.mean(leads)) if leads else 0.0
    return precision, recall, key_acc, mean_lead


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run the REAL anticipatory predictor on a locomotion trace.")
    ap.add_argument("--generator", default="mock")
    ap.add_argument("--horizon-ms", type=float, default=350.0, dest="horizon_ms")
    ap.add_argument("--gen-latency-ms", type=float, default=None, dest="gen_latency_ms")
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prebake", action="store_true")
    args = ap.parse_args(argv)

    trace = simulate(seed=args.seed, duration_s=args.duration)
    preds = AnticipatoryPredictor(horizon_ms=args.horizon_ms).predict(trace)
    gen = get_generator(args.generator)
    pipe = PredictivePipeline(gen, gen_latency_ms=args.gen_latency_ms)
    if args.prebake:
        pipe.prebake(sorted({e.key for e in trace.events}, key=str), variants=4, pin=True)
    res = pipe.run(trace.events, preds)
    s = summarize(res)
    precision, recall, key_acc, mean_lead = match_metrics(preds, trace.events)

    print(f"\n=== REAL PREDICTIVE RUN ({args.duration:.0f}s locomotion, gen={args.generator}, "
          f"horizon={args.horizon_ms:.0f}ms, modeled latency={pipe.gen_latency_ms:.0f}ms"
          f"{', prebake' if args.prebake else ''}) ===")
    print(f" footstep events: {len(trace.events)}    predictions emitted: {len(preds)}")
    print(f" PREDICTOR   precision {precision*100:4.0f}%   recall {recall*100:4.0f}%   "
          f"material+intensity {key_acc*100:4.0f}%   mean lead {mean_lead*1000:4.0f} ms")
    print(f" LOOP        hit-rate {s['hit_rate']*100:4.0f}%   (cached {s['cached']} / fallback {s['fallback']})   "
          f"cold {s['cold_hit_rate']*100:4.0f}%   warm {s['warm_hit_rate']*100:4.0f}%")
    print("\n predictor forecasts from game state alone (speed, gait phase, surface) - no peeking at events.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

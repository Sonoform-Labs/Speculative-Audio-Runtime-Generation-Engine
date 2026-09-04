from __future__ import annotations

import argparse
import os

from .cache import VariantCache
from .generators import get_generator
from .logs import load_events
from .metrics import report, summarize
from .pipeline import PredictivePipeline
from .predictor import NoisyPredictor, Tier1DeterministicPredictor
from .types import CacheKey, Intensity, Material

_DEFAULT_LOG = os.path.join("data", "movement_logs", "sample_run.json")


def _build_predictor(args):
    if args.predictor == "tier1":
        return Tier1DeterministicPredictor(lead_ms=args.lead_ms)
    return NoisyPredictor(
        lead_ms=args.lead_ms, recall=args.recall, key_accuracy=args.accuracy,
        false_pos_rate=args.false_pos, seed=args.seed,
    )


def _run_once(args, events, gen_latency_ms):
    gen = get_generator(args.generator)
    cache = VariantCache(
        max_bytes=(int(args.cache_mb * 1024 * 1024) if args.cache_mb else None),
        rng=__import__("numpy").random.default_rng(args.seed),
    )
    pipe = PredictivePipeline(gen, cache, gen_latency_ms=gen_latency_ms)
    if args.prebake:
        keys = sorted({e.key for e in events}, key=str)
        pipe.prebake(keys, variants=args.prebake_variants, pin=True)
    preds = _build_predictor(args).predict(events)
    return pipe.run(events, preds)


def main_sim(argv=None) -> int:
    p = argparse.ArgumentParser(description="Simulate the predictive neural SFX loop.")
    p.add_argument("--log", default=_DEFAULT_LOG, help="movement log JSON")
    p.add_argument("--generator", default="mock", help="mock | stableaudio")
    p.add_argument("--predictor", default="tier1", choices=["tier1", "noisy"])
    p.add_argument("--lead-ms", type=float, default=250.0, dest="lead_ms",
                   help="prediction lead time (head start before the event)")
    p.add_argument("--gen-latency-ms", type=float, default=None, dest="gen_latency_ms",
                   help="override modeled generation latency (default: generator's)")
    p.add_argument("--recall", type=float, default=0.9, help="noisy predictor: fraction of events predicted")
    p.add_argument("--accuracy", type=float, default=0.85, help="noisy predictor: fraction with correct key")
    p.add_argument("--false-pos", type=float, default=0.1, dest="false_pos",
                   help="noisy predictor: extra wasted guesses per event")
    p.add_argument("--prebake", action="store_true", help="pre-bake the offline (pinned) tier")
    p.add_argument("--prebake-variants", type=int, default=5, dest="prebake_variants")
    p.add_argument("--cache-mb", type=float, default=None, help="runtime cache budget (MB)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sweep", nargs="?", const="latency", default=None,
                   choices=["latency", "accuracy"],
                   help="run a parameter sweep: 'latency' (default) or 'accuracy'")
    args = p.parse_args(argv)

    events = load_events(args.log)
    print(f"Loaded {len(events)} events from {args.log}")
    print(f"Generator={args.generator}  predictor={args.predictor}  lead={args.lead_ms:.0f}ms  "
          f"prebake={'on' if args.prebake else 'off'}\n")

    if args.sweep == "accuracy":
        args.predictor = "noisy"
        print(" Prediction key-accuracy sweep (noisy predictor, "
              f"lead={args.lead_ms:.0f}ms, recall={args.recall}, "
              f"prebake={'on' if args.prebake else 'off'})\n")
        print(f" {'key accuracy':>12} | {'hit rate':>9} | {'fallback':>9} | {'wasted gen':>10}")
        print(" " + "-" * 52)
        for acc in (1.0, 0.9, 0.8, 0.6, 0.4, 0.2):
            args.accuracy = acc
            res = _run_once(args, events, gen_latency_ms=args.gen_latency_ms)
            s = summarize(res)
            print(f" {acc:>12.2f} | {s['hit_rate']*100:>7.1f}% | "
                  f"{s['fallback_rate']*100:>7.1f}% | {s['wasted_generations']:>10}")
        print("\nReads: with caching, hit rate degrades GRACEFULLY with prediction accuracy "
              "(recurring\ncontexts still warm the right slot sometimes). Mispredicted keys waste "
              "generation and\nleave the correct slot cold. This plus context cardinality - not raw "
              "model speed - is the\nsteady-state lever. Add --prebake to watch the authored-baked "
              "floor erase the effect.")
        return 0

    if args.sweep == "latency":
        print(" Generation-latency sweep (predictor lead fixed at "
              f"{args.lead_ms:.0f} ms)\n")
        print(f" {'gen latency':>12} | {'hit rate':>9} | {'cold-start':>10} | {'warm':>7} | {'fallback':>9}")
        print(" " + "-" * 58)
        for lat in (25, 75, 150, 250, 500, 1000, 2000):
            res = _run_once(args, events, gen_latency_ms=float(lat))
            s = summarize(res)
            print(f" {lat:>9} ms | {s['hit_rate']*100:>7.1f}% | {s['cold_hit_rate']*100:>8.1f}% | "
                  f"{s['warm_hit_rate']*100:>5.1f}% | {s['fallback_rate']*100:>7.1f}%")
        print("\nReads: once gen latency exceeds the prediction lead, the COLD-START hit rate "
              "collapses\n(first time each context is seen). The cache stays WARM after that, so "
              "recurring\ncontexts keep hitting. The real steady-state levers are prediction "
              "key-accuracy and\ncontext cardinality (try: --predictor noisy --accuracy 0.6), "
              "not raw latency.")
        return 0

    res = _run_once(args, events, gen_latency_ms=args.gen_latency_ms)
    print(report(res))
    return 0


def main_bake(argv=None) -> int:
    p = argparse.ArgumentParser(description="Bake an offline library of SFX variants.")
    p.add_argument("--generator", default="mock", help="mock | stableaudio")
    p.add_argument("--out", default=os.path.join("data", "baked"), help="output directory")
    p.add_argument("--variants", type=int, default=8, help="variants per key")
    p.add_argument("--event-types", nargs="+", default=["footstep"], dest="event_types")
    p.add_argument("--materials", nargs="+", default=[m.value for m in Material])
    p.add_argument("--intensities", nargs="+", default=[i.value for i in Intensity])
    args = p.parse_args(argv)

    gen = get_generator(args.generator)
    manifest = []
    total = 0
    for et in args.event_types:
        for mat in args.materials:
            for inten in args.intensities:
                key = CacheKey(et, mat, inten)
                for v in range(args.variants):
                    clip = gen.generate(key, variant=v)
                    rel = os.path.join(et, mat, f"{inten}_{v:02d}.wav")
                    dest = os.path.join(args.out, rel)
                    clip.to_wav(dest)
                    manifest.append({"key": str(key), "variant": v, "path": rel,
                                     "duration_s": round(clip.duration_s, 4)})
                    total += 1
    import json
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"generator": gen.name, "count": total, "items": manifest}, f, indent=2)
    print(f"Baked {total} clips with generator '{gen.name}' -> {args.out}")
    return 0

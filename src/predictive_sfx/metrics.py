"""Turn a SimResult into the numbers that decide feasibility."""
from __future__ import annotations

import numpy as np

from .pipeline import SimResult


def summarize(res: SimResult) -> dict:
    n = len(res.outcomes)
    if n == 0:
        return {"events": 0}

    hits = sum(o.hit for o in res.outcomes)
    cached = sum(o.served == "cached" for o in res.outcomes)
    fallback = sum(o.served == "fallback" for o in res.outcomes)
    silence = sum(o.served == "silence" for o in res.outcomes)
    predicted = sum(o.predicted for o in res.outcomes)

    gl_s = res.gen_latency_ms / 1000.0
    leads = [o.lead_s for o in res.outcomes if o.lead_s is not None]
    # Feasibility inequality: among events that had a correct prediction, how many
    # had enough lead time for generation to finish (lead >= gen_latency)?
    in_time = sum(1 for o in res.outcomes if o.lead_s is not None and o.lead_s >= gl_s)

    # Speculation precision: of all clips we generated, how many were actually used?
    precision = ((res.n_generations - res.n_wasted_generations) / res.n_generations
                 if res.n_generations else 0.0)

    # Cold vs warm: the first time a key is requested it may be cold (nothing
    # cached yet); after that the context-keyed cache stays warm, so generation
    # latency only ever costs cold-start misses, not steady-state ones.
    seen: set = set()
    cold = 0
    cold_hits = 0
    for o in res.outcomes:
        k = o.event.key
        if k not in seen:
            seen.add(k)
            cold += 1
            cold_hits += int(o.hit)
    warm = n - cold
    warm_hits = hits - cold_hits

    # Repetition: same variant played twice in a row for the same key.
    last_variant: dict = {}
    immediate_repeats = 0
    for o in res.outcomes:
        if o.variant_id is not None:
            k = o.event.key
            if last_variant.get(k) == o.variant_id:
                immediate_repeats += 1
            last_variant[k] = o.variant_id

    return {
        "events": n,
        "hit_rate": hits / n,
        "cached": cached,
        "fallback": fallback,
        "silence": silence,
        "fallback_rate": fallback / n,
        "predicted_rate": predicted / n,
        "ready_in_time_among_predicted": (in_time / predicted) if predicted else 0.0,
        "unique_keys": cold,
        "cold_hit_rate": (cold_hits / cold) if cold else 0.0,
        "warm_hit_rate": (warm_hits / warm) if warm else 0.0,
        "lead_s_mean": float(np.mean(leads)) if leads else 0.0,
        "lead_s_p50": float(np.median(leads)) if leads else 0.0,
        "gen_latency_ms": res.gen_latency_ms,
        "generations": res.n_generations,
        "wasted_generations": res.n_wasted_generations,
        "speculation_precision": precision,
        "immediate_repeat_rate": immediate_repeats / n,
        "cache_peak_kb": res.cache_peak_bytes / 1024.0,
    }


def report(res: SimResult) -> str:
    s = summarize(res)
    if s.get("events", 0) == 0:
        return "(no events)"

    def pct(x: float) -> str:
        return f"{x * 100:5.1f}%"

    rows = [
        ("Events", f"{s['events']}"),
        ("Hit rate (served from cache)", pct(s["hit_rate"])),
        ("  - cached / fallback / silence", f"{s['cached']} / {s['fallback']} / {s['silence']}"),
        ("Fallback rate (authored)", pct(s["fallback_rate"])),
        ("", ""),
        ("Prediction coverage", pct(s["predicted_rate"])),
        ("Ready-in-time | predicted", pct(s["ready_in_time_among_predicted"])),
        ("Lead time mean / p50", f"{s['lead_s_mean']*1000:.0f} / {s['lead_s_p50']*1000:.0f} ms"),
        ("Gen latency (modeled)", f"{s['gen_latency_ms']:.0f} ms"),
        ("", ""),
        (f"Cold-start hit rate ({s['unique_keys']} keys)", pct(s["cold_hit_rate"])),
        ("Warm (steady-state) hit rate", pct(s["warm_hit_rate"])),
        ("", ""),
        ("Generations (total / wasted)", f"{s['generations']} / {s['wasted_generations']}"),
        ("Speculation precision", pct(s["speculation_precision"])),
        ("Immediate-repeat rate", pct(s["immediate_repeat_rate"])),
        ("Cache peak", f"{s['cache_peak_kb']:.0f} KB"),
    ]
    width = 56
    out = ["=" * width, " PREDICTIVE NEURAL SFX - SIMULATION REPORT", "=" * width]
    for label, val in rows:
        if not label and not val:
            out.append("-" * width)
        else:
            out.append(f" {label:<34}{val:>{width-36}}")
    out.append("=" * width)
    return "\n".join(out)

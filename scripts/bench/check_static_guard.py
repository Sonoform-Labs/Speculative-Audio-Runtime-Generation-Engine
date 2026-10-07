import _bootstrap

import argparse
import json
import os
import time

import numpy as np
import torch

from sarge.generators.stable_audio3 import StableAudio3Generator
from sarge.static_guard import is_static
from sarge.types import CacheKey

WATER_KEYS = [CacheKey("splash", "water", "normal"), CacheKey("footstep", "water", "normal"),
              CacheKey("impact", "water", "heavy"), CacheKey("land", "water", "normal")]
FOOT_KEYS = [CacheKey("footstep", m, "normal") for m in ("wood", "stone", "metal", "grass", "gravel", "dirt")]
IMPACT_KEYS = [CacheKey("impact", m, "normal") for m in ("glass", "metal", "wood", "stone")]
KEYS = WATER_KEYS + FOOT_KEYS + IMPACT_KEYS
PAIRS = [(CacheKey("footstep", "grass", "normal"), CacheKey("impact", "glass", "normal")),
         (CacheKey("footstep", "wood", "normal"), CacheKey("splash", "water", "normal"))]


def timed(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, (time.perf_counter() - t0) * 1000.0


def stat(g):
    return dict(g.static_stats)


def pct(v, q):
    return float(np.percentile(v, q)) if len(v) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--batch-seeds", type=int, default=20)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--out", default=os.path.join("scripts", "bench", "out_static_guard.json"))
    args = ap.parse_args()

    g = StableAudio3Generator(static_guard=False, max_static_retries=args.retries)
    g.ensure_ready()
    sr = int(g.sample_rate)
    print("model %s fp16=%s cfg=%.1f steps=%d detail_steps=%d detail_keys=%s sampler=%s gen=%.1fs keep=%.1fs pad=%.1fs "
          "fast=%s hoist=%s sr=%d retries=%d" % (
              g.model_name, g.model_half, g.cfg_scale, g.steps, g.detail_steps, sorted(g.detail_keys), g.sampler_type,
              g.gen_duration_s, g.duration_s, g.duration_padding_s, g._fast is not None, g._hoist is not None, sr,
              g.max_static_retries))
    for k in KEYS:
        g.generate(k, variant=0)
    for a, b in PAIRS:
        g.generate_batch([(a, 0), (b, 0)])

    off = {}
    for k in KEYS:
        for v in range(args.seeds):
            c, ms = timed(lambda: g.generate(k, variant=v))
            bad, m = is_static(c.samples, sr)
            off[(str(k), v)] = {"bytes": c.samples.tobytes(), "ms": ms, "static": int(bad), "flat": m["flat"],
                                "rule": m["rule"]}
    t_check = []
    for r in list(off.values())[:200]:
        x = np.frombuffer(r["bytes"], dtype=np.float32)
        t0 = time.perf_counter()
        is_static(x, sr)
        t_check.append((time.perf_counter() - t0) * 1000.0)

    g.static_guard = True
    rows = []
    for k in KEYS:
        for v in range(args.seeds):
            s0 = stat(g)
            c, ms = timed(lambda: g.generate(k, variant=v))
            s1 = stat(g)
            retries = s1["static_retries"] - s0["static_retries"]
            exhausted = s1["static_exhausted"] - s0["static_exhausted"]
            bad, m = is_static(c.samples, sr)
            ref = off[(str(k), v)]
            same = c.samples.tobytes() == ref["bytes"]
            rows.append({"key": str(k), "event": k.event_type, "material": k.material, "v": v,
                         "steps": g._steps_for(k), "first_static": ref["static"], "first_rule": ref["rule"],
                         "first_flat": ref["flat"], "retried": int(retries > 0), "attempts": 1 + retries,
                         "exhausted": int(exhausted > 0), "final_static": int(bad), "final_flat": m["flat"],
                         "ms_off": ref["ms"], "ms_on": ms, "identical_to_off": int(same)})

    brows = []
    for a, b in PAIRS:
        for v in range(args.batch_seeds):
            items = [(a, v), (b, v)]
            g.static_guard = False
            ref, ms_off = timed(lambda: g.generate_batch(items))
            g.static_guard = True
            s0 = stat(g)
            out, ms_on = timed(lambda: g.generate_batch(items))
            s1 = stat(g)
            for (k, vv), r, c in zip(items, ref, out):
                rb, _m = is_static(r.samples, sr)
                cb, _m = is_static(c.samples, sr)
                brows.append({"key": str(k), "v": vv, "first_static": int(rb), "final_static": int(cb),
                              "variant_ok": int(c.variant_id == vv),
                              "identical_to_off": int(c.samples.tobytes() == r.samples.tobytes())})
            brows[-1]["batch_retries"] = s1["static_retries"] - s0["static_retries"]
            brows[-1]["batch_exhausted"] = s1["static_exhausted"] - s0["static_exhausted"]
            brows[-1]["ms_off"], brows[-1]["ms_on"] = ms_off, ms_on

    W = 118
    print()
    print("=" * W)
    print(" STATIC GUARD on %d keys x %d seeds (production settings), guard off first, then guard on, same instance" % (
        len(KEYS), args.seeds))
    print("=" * W)
    print("%-30s %5s %6s %7s %7s %6s %6s %6s %9s %9s %9s" % (
        "key", "steps", "first", "retried", "attmax", "exh", "final", "ident", "off ms", "on ms", "retry ms"))
    per = {}
    for k in KEYS:
        rs = [r for r in rows if r["key"] == str(k)]
        clean = [r for r in rs if not r["retried"]]
        rt = [r["ms_on"] for r in rs if r["retried"]]
        p = {"n": len(rs), "first_static": sum(r["first_static"] for r in rs), "retried": sum(r["retried"] for r in rs),
             "attempts_max": max(r["attempts"] for r in rs), "exhausted": sum(r["exhausted"] for r in rs),
             "final_static": sum(r["final_static"] for r in rs),
             "identical_when_no_retry": "%d/%d" % (sum(r["identical_to_off"] for r in clean), len(clean)),
             "ms_off_median": float(np.median([r["ms_off"] for r in rs])),
             "ms_on_median_no_retry": float(np.median([r["ms_on"] for r in clean])) if clean else float("nan"),
             "ms_on_median_retry": float(np.median(rt)) if rt else float("nan")}
        per[str(k)] = p
        print("%-30s %5d %6d %7d %7d %6d %6d %6s %9.1f %9.1f %9.1f" % (
            str(k), g._steps_for(k), p["first_static"], p["retried"], p["attempts_max"], p["exhausted"],
            p["final_static"], p["identical_when_no_retry"], p["ms_off_median"], p["ms_on_median_no_retry"],
            p["ms_on_median_retry"]))
    print("-" * W)
    clean = [r for r in rows if not r["retried"]]
    retried = [r for r in rows if r["retried"]]
    tot = {"clips": len(rows), "first_static": sum(r["first_static"] for r in rows), "retried": len(retried),
           "exhausted": sum(r["exhausted"] for r in rows), "final_static": sum(r["final_static"] for r in rows),
           "final_static_not_exhausted": sum(r["final_static"] for r in rows if not r["exhausted"]),
           "identical_when_no_retry": sum(r["identical_to_off"] for r in clean), "no_retry": len(clean),
           "retried_without_first_static": sum(1 for r in retried if not r["first_static"]),
           "first_static_without_retry": sum(1 for r in rows if r["first_static"] and not r["retried"]),
           "ms_off_median": float(np.median([r["ms_off"] for r in rows])),
           "ms_off_p95": pct([r["ms_off"] for r in rows], 95),
           "ms_on_no_retry_median": float(np.median([r["ms_on"] for r in clean])) if clean else float("nan"),
           "ms_on_no_retry_p95": pct([r["ms_on"] for r in clean], 95),
           "ms_on_retry_median": float(np.median([r["ms_on"] for r in retried])) if retried else float("nan"),
           "ms_on_retry_max": float(max([r["ms_on"] for r in retried])) if retried else float("nan"),
           "check_ms_median": float(np.median(t_check)), "check_ms_p95": pct(t_check, 95)}
    print(" total %d clips: first-attempt static %d, retried %d, exhausted %d, final static %d (%d outside exhausted)" % (
        tot["clips"], tot["first_static"], tot["retried"], tot["exhausted"], tot["final_static"],
        tot["final_static_not_exhausted"]))
    print(" guard-on output byte-identical to guard-off on %d/%d clips with no retry; retried without a static first "
          "attempt %d; static first attempt left unretried %d" % (
              tot["identical_when_no_retry"], tot["no_retry"], tot["retried_without_first_static"],
              tot["first_static_without_retry"]))
    print(" latency ms: guard off median %.1f p95 %.1f | guard on, no retry median %.1f p95 %.1f | with retry median "
          "%.1f max %.1f | is_static on a %.1f s clip median %.3f p95 %.3f" % (
              tot["ms_off_median"], tot["ms_off_p95"], tot["ms_on_no_retry_median"], tot["ms_on_no_retry_p95"],
              tot["ms_on_retry_median"], tot["ms_on_retry_max"], g.duration_s, tot["check_ms_median"],
              tot["check_ms_p95"]))
    bt = {"items": len(brows), "first_static": sum(r["first_static"] for r in brows),
          "final_static": sum(r["final_static"] for r in brows),
          "variant_order_ok": sum(r["variant_ok"] for r in brows),
          "identical_when_first_clean": "%d/%d" % (sum(r["identical_to_off"] for r in brows if not r["first_static"]),
                                                   sum(1 for r in brows if not r["first_static"])),
          "retries": sum(r.get("batch_retries", 0) for r in brows),
          "exhausted": sum(r.get("batch_exhausted", 0) for r in brows)}
    print(" batch pairs %s x %d seeds: items %d, first static %d, final static %d, order ok %d, identical when first "
          "clean %s, retries %d, exhausted %d" % (
              "+".join("%s/%s" % (a.material, b.material) for a, b in PAIRS), args.batch_seeds, bt["items"],
              bt["first_static"], bt["final_static"], bt["variant_order_ok"], bt["identical_when_first_clean"],
              bt["retries"], bt["exhausted"]))
    fs = g.fast_stats()
    print(" generator stats %s" % json.dumps(fs))
    print("=" * W)
    clean_b = [r for r in brows if not r["first_static"]]
    verdict = (tot["final_static_not_exhausted"] == 0 and tot["identical_when_no_retry"] == tot["no_retry"]
               and tot["retried_without_first_static"] == 0 and tot["first_static_without_retry"] == 0
               and bt["variant_order_ok"] == bt["items"] and bt["final_static"] <= bt["exhausted"]
               and all(r["identical_to_off"] for r in clean_b))
    print(" VERDICT %s" % ("PASS" if verdict else "CHECK"))
    json.dump({"settings": {"model": g.model_name, "steps": g.steps, "detail_steps": g.detail_steps,
                            "detail_keys": sorted(g.detail_keys), "sampler": g.sampler_type, "cfg": g.cfg_scale,
                            "gen_s": g.gen_duration_s, "keep_s": g.duration_s, "pad_s": g.duration_padding_s,
                            "retries": g.max_static_retries, "seeds": args.seeds},
               "total": tot, "per_key": per, "batch": bt, "fast_stats": fs, "verdict": bool(verdict),
               "rows": rows, "batch_rows": brows},
              open(args.out, "w"), indent=1)
    print("saved -> %s" % args.out)


if __name__ == "__main__":
    main()

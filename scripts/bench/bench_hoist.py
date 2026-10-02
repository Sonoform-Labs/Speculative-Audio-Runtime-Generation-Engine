import os
import sys


def find_root():
    for start in (os.getcwd(), os.path.dirname(os.path.abspath(__file__))):
        p = os.path.abspath(start)
        while True:
            if os.path.isfile(os.path.join(p, "pyproject.toml")) and os.path.isdir(os.path.join(p, "src", "sarge")):
                return p
            q = os.path.dirname(p)
            if q == p:
                break
            p = q
    return None


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = find_root()
if ROOT is None:
    sys.exit("run from the repo root (D:\\Python\\SAGE), cwd=%s" % os.getcwd())
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "bench"))
if HERE not in sys.path:
    sys.path.insert(1, HERE)

import _bootstrap

import argparse
import gc
import json
import time

import numpy as np
import torch

from sarge.generators.stable_audio import _seed
from sarge.generators.stable_audio3 import StableAudio3Generator
from sarge.types import CacheKey

try:
    from sarge.generators import sa3_hoist as hoist_mod
except ImportError:
    import hoist as hoist_mod
try:
    from sarge.generators import decode_prefix as prefix_mod
except ImportError:
    import decode_prefix as prefix_mod

KEYS = [CacheKey("footstep", "wood", "normal"), CacheKey("footstep", "metal", "heavy"),
        CacheKey("impact", "glass", "normal"), CacheKey("footstep", "water", "normal")]
PAIRS = [[(CacheKey("footstep", "stone", "normal"), 3), (CacheKey("footstep", "grass", "normal"), 4)],
         [(CacheKey("impact", "metal", "normal"), 1), (CacheKey("footstep", "wood", "light"), 2)]]
CONFIGS = [("cond", ("cond",), 0), ("cond+kv", ("cond", "kv"), 0), ("local", ("local",), 0),
           ("rope", ("rope",), 0), ("default+rope", hoist_mod.DEFAULT_PARTS + ("rope",), 0),
           ("default+cache8", hoist_mod.DEFAULT_PARTS, 8)]


def timed(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, (time.perf_counter() - t0) * 1000.0


def med(xs):
    return float(np.median(xs)) if xs else float("nan")


def jobs(g):
    js = [(str(k), (lambda k=k: [g.generate(k, variant=0)])) for k in KEYS]
    js += [("batch2 %s+%s" % (p[0][0].material, p[1][0].material), (lambda p=p: g.generate_batch(p)))
           for p in PAIRS]
    return js


def run(g, reps):
    res = {}
    for name, job in jobs(g):
        job()
        ts, out = [], None
        for _ in range(reps):
            cs, ms = timed(job)
            ts.append(ms)
            if out is None:
                out = [c.samples for c in cs]
        res[name] = {"ms": med(ts), "out": out}
    return res


def run_paired(g, toggle, reps):
    res = {}
    for name, job in jobs(g):
        for on in (False, True):
            toggle(on)
            job()
        ta, tb, oa, ob = [], [], None, None
        for r in range(reps):
            for on in ((False, True) if r % 2 == 0 else (True, False)):
                toggle(on)
                cs, ms = timed(job)
                if on:
                    tb.append(ms)
                    if ob is None:
                        ob = [c.samples for c in cs]
                else:
                    ta.append(ms)
                    if oa is None:
                        oa = [c.samples for c in cs]
        res[name] = {"a_ms": med(ta), "b_ms": med(tb), "a": oa, "b": ob}
    toggle(True)
    return res


def diff(a, b):
    if a is None or b is None or len(a) != len(b):
        return float("inf"), False
    ds = [float(np.max(np.abs(x - y))) if x.shape == y.shape else float("inf") for x, y in zip(a, b)]
    ex = all(x.shape == y.shape and np.array_equal(x, y) for x, y in zip(a, b))
    return (max(ds) if ds else float("nan")), ex


def table(title, labels, ref, paired, again):
    print()
    print(title)
    print("%-34s %9s %9s %9s %12s %6s" % ("key", labels[0], labels[1], "saved", "max|diff|", "exact"))
    rows, ok, rep_all, sane_all = {}, True, True, True
    for name, p in paired.items():
        base = ref[name]["out"] if ref is not None else p["a"]
        d, ex = diff(base, p["b"])
        rep = diff(p["b"], again[name]["out"])[1] if again is not None else None
        sane = diff(ref[name]["out"], p["a"])[1] if ref is not None else None
        print("%-34s %9.1f %9.1f %9.1f %12.2e %6s" % (name, p["a_ms"], p["b_ms"], p["a_ms"] - p["b_ms"], d, ex))
        rows[name] = {"a_ms": p["a_ms"], "b_ms": p["b_ms"], "saved_ms": p["a_ms"] - p["b_ms"],
                      "max_abs_diff": d, "exact": ex, "repeatable": rep, "off_matches_ref": sane}
        ok = ok and ex
        rep_all = rep_all and rep is not False
        sane_all = sane_all and sane is not False
    print("repeatable across reruns: %s   off path matches pre-install ref: %s" % (rep_all, sane_all))
    return ok and rep_all and sane_all, rows


def alternate(g, rounds):
    res = {}
    js = jobs(g)
    for _ in range(rounds):
        for name, job in js:
            res.setdefault(name, []).append([c.samples for c in job()])
    return res


def release(*objs):
    for o in objs:
        if o is not None:
            o.uninstall()
    gc.collect()
    torch.cuda.empty_cache()


def trial(g, title, parts, cache, args, ref):
    torch.cuda.reset_peak_memory_stats()
    h = hoist_mod.install_hoist(g._model, g._fast, parts=parts, graph_fill=not args.eager_fill,
                                cache_prompts=cache)
    if not h.installed:
        print()
        print("%s: hoist skipped: %s" % (title, h.reason))
        st = h.stats()
        release(h)
        return {"installed": False, "stats": st}, False
    paired = run_paired(g, lambda on: setattr(h, "enabled", on), args.reps)
    again = run(g, 1)
    ok, rows = table("%s   parts=%s cache=%d" % (title, "+".join(sorted(h.parts)), cache),
                     ("fast ms", "hoist ms"), ref, paired, again)
    alt = alternate(g, 2)
    alt_ok = all(diff(ref[n]["out"], o)[1] for n, outs in alt.items() for o in outs)
    print("prompt alternation matches pre-install ref: %s" % alt_ok)
    h.verify = True
    v0 = dict(h.counts)
    vrun = run(g, 1)
    h.verify = False
    viol = h.counts["violations"] - v0["violations"]
    mism = h.counts["cache_mismatch"] - v0["cache_mismatch"]
    vsame = all(diff(paired[n]["b"], vrun[n]["out"])[1] for n in paired)
    print("invariance check: violations %d   cache mismatches %d   verify run matches: %s" % (viol, mism, vsame))
    st = h.stats()
    print("hoist stats", {k: v for k, v in st.items() if k not in ("errors",)})
    if st["errors"]:
        print("hoist errors", st["errors"])
    mem = torch.cuda.max_memory_allocated() / 1e9
    print("peak mem %.2f GB" % mem)
    release(h)
    good = ok and alt_ok and viol == 0 and mism == 0 and vsame
    return {"installed": True, "rows": rows, "exact": ok, "alternation_matches": alt_ok, "violations": viol,
            "cache_mismatch": mism, "verify_matches": vsame, "stats": st, "peak_mem_gb": mem, "pass": good}, good


def decode_probe(g, reps):
    sam = g._model
    pt = sam.model.pretransform
    keep = int(g.gen_duration_s * sam.model.sample_rate)
    res = {}
    for k in KEYS[:3]:
        with torch.no_grad():
            lat = sam.generate(duration=g.gen_duration_s, steps=g._steps_for(k), cfg_scale=g.cfg_scale,
                               seed=_seed(k, 0), return_latents=True, **g._inputs([g._prompt(k)]))
        lat = lat.to(next(pt.parameters()).dtype)
        n = prefix_mod.prefix_len(keep, lat.shape[-1], pt.downsampling_ratio, 3, prefix_mod.decoder_align(sam))
        seed = _seed(k, 0)

        def full():
            torch.manual_seed(seed)
            return pt.decode(lat)

        def pre():
            torch.manual_seed(seed)
            return pt.decode(lat[..., :n])

        with torch.inference_mode():
            full()
            pre()
            tf = med([timed(full)[1] for _ in range(reps)])
            tp = med([timed(pre)[1] for _ in range(reps)])
            a = full()[..., :keep].float()
            b = pre()[..., :keep].float()
            d = float((a - b).abs().max().item())
        res[str(k)] = {"latent_len": int(lat.shape[-1]), "prefix_len": int(n), "decode_full_ms": tf,
                       "decode_prefix_ms": tp, "max_abs_diff_first_keep": d}
        print("  decode %-30s latents %d -> %d   full %6.2f ms   prefix %6.2f ms   max|diff| first %d samples %.2e"
              % (str(k), int(lat.shape[-1]), int(n), tf, tp, keep, d))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--eager-fill", action="store_true")
    ap.add_argument("--no-ablation", action="store_true")
    ap.add_argument("--no-prefix", action="store_true")
    ap.add_argument("--out", default=os.path.join("scripts", "bench", "out_bench_hoist.json"))
    args = ap.parse_args()
    if not torch.cuda.is_available():
        print("cuda not available")
        sys.exit(2)

    g = StableAudio3Generator()
    g.ensure_ready()
    if g._fast is None or getattr(g._fast, "graph", None) is None:
        print("fast path not active, nothing to compare")
        sys.exit(2)
    sam = g._model
    ds = sam.model.pretransform.downsampling_ratio
    n = sam._adapt_sample_size([{"prompt": "x", "seconds_total": g.gen_duration_s}], 5292032, g.duration_padding_s)
    facts = {"torch": torch.__version__, "device": torch.cuda.get_device_name(0), "steps": g.steps,
             "detail_steps": g.detail_steps, "cfg": g.cfg_scale, "sampler": g.sampler_type,
             "padding_s": g.duration_padding_s, "latent_len": int(n // ds), "reps": args.reps,
             "graph_fill": not args.eager_fill, "hoist_module": hoist_mod.__file__,
             "prefix_module": prefix_mod.__file__}
    for k, v in facts.items():
        print("%-16s %s" % (k, v))

    ref = run(g, args.reps)
    print()
    print("pre-install fast path")
    for name, r in ref.items():
        print("%-34s %9.1f ms" % (name, r["ms"]))

    res = {"facts": facts, "ref_ms": {k: v["ms"] for k, v in ref.items()}, "configs": {}}
    out, good = trial(g, "default", hoist_mod.DEFAULT_PARTS, 0, args, ref)
    res["configs"]["default"] = out
    res["default_pass"] = good

    if not args.no_ablation:
        for title, parts, cache in CONFIGS:
            out, _ = trial(g, title, parts, cache, args, ref)
            res["configs"][title] = out

    if not args.no_prefix:
        print()
        print("decode prefix, decode only on fast path latents")
        res["decode_probe"] = decode_probe(g, args.reps)
        h = hoist_mod.install_hoist(sam, g._fast, parts=hoist_mod.DEFAULT_PARTS, graph_fill=not args.eager_fill)
        dp = prefix_mod.install_decode_prefix(sam)
        if h.installed and dp.installed:
            paired = run_paired(g, lambda on: setattr(dp, "enabled", on), args.reps)
            again = run(g, 1)
            _, rows = table("decode prefix on the kept %.1f s clip, on top of hoist default" % g.duration_s,
                            ("hoist ms", "+prefix"), None, paired, again)
            res["prefix"] = {"rows": rows, "stats": dp.stats()}
            print("prefix stats", dp.stats())
        else:
            print("prefix skipped: hoist %s / prefix %s" % (h.reason, dp.reason))
            res["prefix"] = {"skipped": [h.reason, dp.reason]}
        release(dp, h)

    res["accelerator"] = g.fast_stats()
    print()
    print("accelerator", res["accelerator"])
    print("default hoist gate: %s" % ("PASS" if res["default_pass"] else "FAIL"))
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=1, default=str)
    print("saved -> %s" % args.out)
    sys.exit(0 if res["default_pass"] else 1)


if __name__ == "__main__":
    main()

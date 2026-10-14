import argparse
import ast
import glob
import hashlib
import html
import itertools
import json
import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

import woosh_opt as wo

OPT = ROOT / "data" / "_gate" / "opt" / "medium"
WORK = OPT / "_work"
RAW, FIN, ROUTES, LISTEN = (WORK / d for d in ("raw", "fin", "routes", "listen"))
INTENSITY, CONTEXT = "normal", "default"
GK = [("shoot", "pistol"), ("shoot", "rifle"), ("reload", "rifle"), ("empty_click", "pistol"), ("equip", "rifle")]
WARM_KEY = ("explosion", "grenade")
G2_SEEDS = 4
FINISH = {"amp": True, "stereo": True, "trim": "natural", "level": "peak"}
SR = 44100
SHAPES = {"m_qkv": (1536, 7680, False), "m_proj": (1536, 1536, False), "m_q": (1536, 3072, False),
          "m_kv": (1536, 4608, False), "m_ff_in": (1536, 12288, True), "m_ff_out": (6144, 1536, True)}
PER_STEP = {"m_qkv": 24, "m_proj": 48, "m_q": 24, "m_kv": 24, "m_ff_in": 24, "m_ff_out": 24}
KV_M = 257
CFG_ATTRS = ("model_name", "sampler_type", "steps", "detail_steps", "cfg_scale", "gen_duration_s", "duration_padding_s",
             "skip_first_step", "compile_mode", "compile_decode", "graph_decode", "hoist_rope", "fp16_accumulation",
             "graph_text", "unfuse_bias", "triton_gemm", "decode_prefix", "decode_margin_frames", "prompt_style",
             "static_guard", "model_half", "fast", "trim", "level", "amp", "stereo", "noise_span", "hoist_diff",
             "hoist_prompts")
FAIL_STATS = ("capture_failures", "decode_failures", "text_failures")
TAIL_MAX = 1

wo.OPT, wo.WORK, wo.RAW, wo.FIN, wo.ROUTES, wo.LISTEN = OPT, WORK, RAW, FIN, ROUTES, LISTEN
wo.WK = list(GK)
wo.J4_POOL_KEYS = ["shoot|pistol", "shoot|rifle"]
wo.J4_ART_KEYS = ["reload|rifle"]
kname, fname, dump, load_json, med, pct = wo.kname, wo.fname, wo.dump, wo.load_json, wo.med, wo.pct


def lit(v):
    try:
        return ast.literal_eval(v)
    except (ValueError, SyntaxError):
        return v


def base_kwargs(base):
    if base == "hybrid":
        sys.path.insert(1, str(ROOT / "scripts"))
        import run_sidecar
        kw = dict(run_sidecar.TIERS["default"], fast_load=True)
        kw.update(run_sidecar.HYBRID_GUNS)
        return kw
    import bakeoff
    spec = bakeoff.SA3_ARMS["medium_rec"]
    return dict(spec["kw"], fast_load=True, prompt_style="detailed", model_name=spec["model"],
                gen_duration_s=spec["gen_s"])


def patch_triton(path, names):
    from sarge.generators import sa3_gemm
    rep = load_json(path)
    keep = names or [n for n, r in rep["shapes"].items() if r["keep"]]
    shapes = []
    for n in keep:
        r = rep["shapes"][n]
        sa3_gemm.CONFIGS[(r["k"], r["n"])] = tuple(r["plain_cfg"])
        shapes.append((r["k"], r["n"]))
    sa3_gemm.DEFAULT_SHAPES = tuple(sa3_gemm.DEFAULT_SHAPES) + tuple(s for s in shapes
                                                                     if s not in sa3_gemm.DEFAULT_SHAPES)
    return {"json": str(path), "shapes": {n: rep["shapes"][n]["plain_cfg"] for n in keep}}


def dyn(counters):
    return {k: {str(kk): vv for kk, vv in v.items()} for k, v in counters.items()}


def graph_error(fs):
    bad = ["%s %s" % (k, fs[k]) for k in FAIL_STATS if fs.get(k)]
    for k in ("hoist_capture_failures", "slot_failures", "errors"):
        if fs.get(k):
            bad.append("%s %s" % (k, str(fs[k])[:200]))
    if fs.get("hoist_eager"):
        bad.append("hoist_eager %s" % fs["hoist_eager"])
    return "; ".join(bad) or None


def noise_check(torch):
    from sarge.generators.stable_audio3 import _SpanTorch
    x = torch.empty(1, 256, 98, device="cuda", dtype=torch.float16)
    torch.manual_seed(5)
    full = torch.randn_like(x)
    torch.manual_seed(5)
    part = _SpanTorch(torch, 98).randn_like(torch.empty(1, 256, 44, device="cuda", dtype=torch.float16))
    torch.manual_seed(5)
    full0 = torch.randn([1, 256, 98], device="cuda")
    torch.manual_seed(5)
    part0 = _SpanTorch(torch, 98).randn([1, 256, 44], device="cuda")
    return {"randn_like_crop_equal": bool(torch.equal(full[..., :44], part)),
            "randn_crop_equal": bool(torch.equal(full0[..., :44], part0))}


def natural_end(x, sr=SR, floor_db=-55.0, fade_s=0.03):
    m = x.mean(axis=1) if x.ndim == 2 else x
    n = m.shape[0]
    win = max(1, int(0.01 * sr))
    if n <= win or not np.any(m):
        return n / sr
    c = np.empty(n + 1)
    c[0] = 0.0
    np.cumsum(m.astype(np.float64) ** 2, out=c[1:])
    env = np.sqrt(np.maximum(c[win:] - c[:-win], 0.0) / win)
    alive = np.flatnonzero(env >= float(env.max()) * 10.0 ** (floor_db / 20.0))
    return min(n, int(alive[-1]) + win + int(fade_s * sr)) / sr


def make_tap():
    import bakeoff

    class TimedTap(bakeoff.Tap):
        def __init__(self, g):
            super().__init__(g)
            self.ms = 0.0

        def __call__(self, a, key):
            t = time.perf_counter()
            out = super().__call__(a, key)
            self.ms += (time.perf_counter() - t) * 1000.0
            return out

    return TimedTap


class DecodeTimer:
    def __init__(self, torch, pt):
        self.torch, self.pt = torch, pt
        self.prev = pt.__dict__.get("decode")
        self.orig = pt.decode
        self.t0 = self.t1 = None
        pt.decode = self

    def __call__(self, z, chunked=None, **kw):
        self.torch.cuda.synchronize()
        self.t0 = time.perf_counter()
        out = self.orig(z, chunked=chunked, **kw)
        self.torch.cuda.synchronize()
        self.t1 = time.perf_counter()
        return out

    def close(self):
        if self.prev is None:
            del self.pt.decode
        else:
            self.pt.decode = self.prev


def cmd_run(a):
    import torch
    from torch._dynamo.utils import counters
    from sarge.generators.stable_audio3 import StableAudio3Generator
    from sarge.types import AudioClip, CacheKey
    for d in (RAW / a.arm, FIN / a.arm, OPT / a.arm):
        if d.is_dir():
            shutil.rmtree(d)
    for d in (RAW / a.arm, FIN / a.arm):
        d.mkdir(parents=True)
    kw = base_kwargs(a.base)
    for s in a.set or []:
        k, v = s.split("=", 1)
        kw[k] = lit(v)
    kw.update(FINISH)
    if a.noise_span:
        kw["noise_span"] = tuple(float(x) for x in a.noise_span.split(","))
    if a.hoist_diff:
        kw["hoist_diff"] = True
    tri = patch_triton(a.triton_medium, a.triton_shapes) if a.triton_medium else None
    t0 = time.perf_counter()
    g = StableAudio3Generator(**kw)
    g.ensure_ready()
    torch.cuda.synchronize()
    load_s = time.perf_counter() - t0
    load_peak = {"peak_alloc_gb": torch.cuda.max_memory_allocated() / 2 ** 30,
                 "peak_reserved_gb": torch.cuda.max_memory_reserved() / 2 ** 30}
    torch.cuda.reset_peak_memory_stats()
    print("loaded medium in %.1f s, peak %s, accelerator %s, hoist %s" % (
        load_s, json.dumps(load_peak), g._fast is not None,
        json.dumps({k: v for k, v in g.fast_stats().items() if k in ("installed", "parts", "reason")})), flush=True)
    tap = make_tap()(g)
    n_gen, n_steps = [0], [0]

    def gen(key, v):
        n_gen[0] += 1
        n_steps[0] += g._steps_for(key)
        return g.generate(key, variant=v)

    t1 = time.perf_counter()
    gen(CacheKey(*WARM_KEY, INTENSITY, CONTEXT), 99)
    tap.raw.clear()
    torch.cuda.synchronize()
    warm_s = time.perf_counter() - t1
    dyn_warm = dyn(counters)
    st_warm = dict(g.fast_stats())
    print("warm-up %.1f s, dynamo %s" % (warm_s, json.dumps(dyn_warm.get("stats", {}))), flush=True)
    keys = list(GK)
    prompts, saved, calls, bad = {}, {}, [], []
    cache = g._fast.cache if g._fast is not None else None
    for et, mat in keys:
        key = CacheKey(et, mat, INTENSITY, CONTEXT)
        prompts[kname(et, mat)] = g._prompt(key)
        ks = []
        for v in range(a.seeds):
            miss0 = cache.misses if cache else 0
            tap.ms = 0.0
            torch.cuda.synchronize()
            t = time.perf_counter()
            clip = gen(key, v)
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t) * 1000.0
            x = np.ascontiguousarray(tap.take(clip.samples), dtype=np.float32)
            y = np.ascontiguousarray(clip.samples, dtype=np.float32)
            if not (np.all(np.isfinite(x)) and np.any(np.abs(x) > 1e-6)):
                bad.append(fname(et, mat, v))
            np.save(RAW / a.arm / (fname(et, mat, v) + ".npy"), x)
            np.save(FIN / a.arm / (fname(et, mat, v) + ".npy"), y)
            saved[fname(et, mat, v)] = x
            calls.append({"pass": 0, "key": kname(et, mat), "v": v, "ms": ms, "finish_ms": tap.ms,
                          "miss": bool(cache is not None and cache.misses > miss0)})
            if a.g2 and v < G2_SEEDS:
                AudioClip(y, g.sample_rate).to_wav(str(OPT / a.arm / (fname(et, mat, v) + ".wav")))
            ks.append(ms)
        print("%-10s %-20s median %.1f ms (%d seeds)" % (a.arm, kname(et, mat), med(ks), a.seeds), flush=True)
    rep_diff = []
    for rep in range(a.reps):
        for et, mat in keys:
            key = CacheKey(et, mat, INTENSITY, CONTEXT)
            for v in range(a.seeds):
                miss0 = cache.misses if cache else 0
                tap.ms = 0.0
                torch.cuda.synchronize()
                t = time.perf_counter()
                clip = gen(key, v)
                torch.cuda.synchronize()
                ms = (time.perf_counter() - t) * 1000.0
                x = np.asarray(tap.take(clip.samples), dtype=np.float32)
                calls.append({"pass": rep + 1, "key": kname(et, mat), "v": v, "ms": ms, "finish_ms": tap.ms,
                              "miss": bool(cache is not None and cache.misses > miss0)})
                ref = saved[fname(et, mat, v)]
                rep_diff.append(float(np.max(np.abs(x - ref))) if x.shape == ref.shape else float("inf"))
    stages = None
    if a.stages:
        dt = DecodeTimer(torch, g._model.model.pretransform)
        rows = []
        try:
            for et, mat in keys:
                key = CacheKey(et, mat, INTENSITY, CONTEXT)
                for v in range(a.stage_seeds):
                    tap.ms = 0.0
                    torch.cuda.synchronize()
                    t = time.perf_counter()
                    clip = gen(key, v)
                    end = time.perf_counter()
                    tap.take(clip.samples)
                    rows.append({"sample": (dt.t0 - t) * 1000.0, "decode": (dt.t1 - dt.t0) * 1000.0,
                                 "tail": (end - dt.t1) * 1000.0 - tap.ms, "finish": tap.ms,
                                 "total": (end - t) * 1000.0})
        finally:
            dt.close()
        stages = {n: med([r[n] for r in rows]) for n in rows[0]}
        print("stages (median ms): %s" % json.dumps({k: round(v, 2) for k, v in stages.items()}), flush=True)
    peak = {"peak_alloc_gb": torch.cuda.max_memory_allocated() / 2 ** 30,
            "peak_reserved_gb": torch.cuda.max_memory_reserved() / 2 ** 30}
    fs = g.fast_stats()
    dyn_end = dyn(counters)
    all_ms = [c["ms"] for c in calls]
    hit = [c["ms"] for c in calls if not c["miss"]]
    miss = [c["ms"] for c in calls if c["miss"]]
    gpu_ms = [c["ms"] - c["finish_ms"] for c in calls]
    timing = {"median_ms": med(all_ms), "p90_ms": pct(all_ms, 90), "p10_ms": pct(all_ms, 10), "n": len(all_ms),
              "save_pass_median_ms": med([c["ms"] for c in calls if c["pass"] == 0]),
              "cond_hit_median_ms": med(hit), "cond_miss_median_ms": med(miss), "n_hit": len(hit),
              "n_miss": len(miss), "finish_median_ms": med([c["finish_ms"] for c in calls]),
              "model_median_ms": med(gpu_ms), "stages_ms": stages,
              "rep_max_abs_diff": max(rep_diff) if rep_diff else None}
    uw = (dyn_warm.get("stats") or {}).get("unique_graphs", 0)
    ue = (dyn_end.get("stats") or {}).get("unique_graphs", 0)
    retries = int(fs.get("static_retries", 0))
    gens = n_gen[0] + retries
    h1 = {"accelerator": g._fast is not None, "hoist_installed": fs.get("installed"), "hoist_parts": fs.get("parts"),
          "generate_calls": gens, "dit_steps": n_steps[0] + retries * g.steps, "hoisted": fs.get("hoisted"),
          "plain": fs.get("plain"), "hoist_eager": fs.get("hoist_eager"),
          "capture_failures": {k: fs.get(k) for k in FAIL_STATS},
          "decode_graphs": fs.get("decode_graphs"), "decode_replays": fs.get("decode_replays"),
          "prefix_decodes": fs.get("prefix_decodes"), "cond_misses": fs.get("cond_misses"),
          "distinct_prompts": len(set(prompts.values())) + 1, "triton_linears": fs.get("triton_linears"),
          "text_graphs": fs.get("text_graphs"), "unique_graphs_after_warm": ue - uw,
          "graph_breaks": sum((dyn_end.get("graph_break") or {}).values())}
    h1["pass"] = bool(h1["hoist_installed"] and h1["hoisted"] == h1["dit_steps"] and not h1["plain"]
                      and not h1["hoist_eager"] and not any(h1["capture_failures"].values())
                      and (not g.graph_decode or (h1["decode_graphs"] or 0) >= 1)
                      and (not g.graph_decode or h1["decode_replays"] == gens)
                      and (not g.decode_prefix or h1["prefix_decodes"] == gens)
                      and h1["cond_misses"] == h1["distinct_prompts"] and h1["unique_graphs_after_warm"] == 0)
    cfg = {k: getattr(g, k, None) for k in CFG_ATTRS}
    cfg["detail_keys"] = sorted(g.detail_keys)
    cfg["triton"] = tri
    cfg["base"] = a.base
    cfg["set"] = a.set or []
    info = {"arm": a.arm, "family": "sa3", "model": "sa3-medium", "gate_class": a.gate, "base": a.vs,
            "note": a.note or "sa3 medium %s %s" % (a.base, " ".join(a.set or [])), "config": cfg, "sr": g.sample_rate,
            "channels": int(next(iter(saved.values())).shape[1]), "seeds": list(range(a.seeds)),
            "keys": [kname(et, mat) for et, mat in keys], "prompts": prompts, "intensity": INTENSITY,
            "context": CONTEXT, "load_s": load_s, "warm_s": warm_s, "median_s": timing["median_ms"] / 1000.0,
            "load_peak": load_peak, **peak, "timing": timing, "calls": calls, "bad_clips": bad,
            "g2_keys": [kname(et, mat) for et, mat in keys] if a.g2 else [], "g2_seeds": G2_SEEDS if a.g2 else 0,
            "static_guard": {k: int(v) for k, v in g.static_stats.items()}, "fast_stats": fs,
            "fast_stats_warm": st_warm, "stats": {"graph_error": graph_error(fs)}, "h1": h1,
            "dynamo_warm": dyn_warm, "dynamo_end": dyn_end,
            "noise_check": noise_check(torch) if g.noise_span else None,
            "torch": torch.__version__, "device": torch.cuda.get_device_name(0),
            "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    dump(info, RAW / a.arm / "gen.json")
    print("%-10s done: median %.1f ms p90 %.1f ms (hit %.1f, miss %s, finish %.1f), warm-up %.1f s, peak %.2f GB alloc "
          "%.2f GB reserved, rep diff %s, h1 %s, graph %s" % (
              a.arm, timing["median_ms"], timing["p90_ms"], timing["cond_hit_median_ms"] or float("nan"),
              "%.1f" % timing["cond_miss_median_ms"] if timing["cond_miss_median_ms"] else "-",
              timing["finish_median_ms"], warm_s, peak["peak_alloc_gb"], peak["peak_reserved_gb"],
              timing["rep_max_abs_diff"], "pass" if h1["pass"] else json.dumps(h1), info["stats"]["graph_error"] or "ok"),
          flush=True)
    if bad:
        print("%-10s %d clips non-finite or silent: %s" % (a.arm, len(bad), " ".join(bad[:8])), flush=True)
    return 0


def cmd_tune(a):
    import torch
    import triton_tune as tt
    from sarge.generators import sa3_gemm
    torch.backends.cuda.matmul.allow_fp16_accumulation = True
    torch.manual_seed(20261014)
    op = sa3_gemm._build()
    steps = a.steps
    rep = {"m": a.m, "kv_m": KV_M, "steps": steps, "kv_per_clip": a.kv_per_clip, "variants": a.variants,
           "shapes": {}}
    for name, (k, n, has_bias) in SHAPES.items():
        m = KV_M if name == "m_kv" else a.m
        x = torch.randn(m, k, device="cuda", dtype=torch.float16)
        w = torch.randn(n, k, device="cuda", dtype=torch.float16) * 0.02
        bias = (torch.randn(n, device="cuda", dtype=torch.float16) * 0.1) if has_bias else None
        ref = torch.nn.functional.linear(x.float(), w.float(), None if bias is None else bias.float())
        t_cub = tt.bench(lambda: torch.nn.functional.linear(x, w, bias))
        err_c = float((torch.nn.functional.linear(x, w, bias).float() - ref).abs().max())
        prev = sa3_gemm.CONFIGS.get((k, n))
        var = {}
        for v in a.variants:
            res = []
            for cfg in itertools.product((32, 64, 128), (64, 128, 256), (32, 64), (4, 8), (3, 4, 5)):
                if cfg[0] * cfg[1] > 128 * 128:
                    continue
                c = cfg if v == "f16" else cfg + ("f32",)
                sa3_gemm.CONFIGS[(k, n)] = c
                try:
                    op(x, w, bias)
                    res.append((tt.bench(lambda: op(x, w, bias)), c))
                except Exception:
                    continue
            res.sort()
            sa3_gemm.CONFIGS[(k, n)] = res[0][1]
            err = float((op(x, w, bias).float() - ref).abs().max())
            var[v] = {"us": round(res[0][0], 2), "cfg": list(res[0][1]), "err": err,
                      "top5": [(round(t, 2), list(c)) for t, c in res[:5]]}
        if prev is None:
            sa3_gemm.CONFIGS.pop((k, n), None)
        else:
            sa3_gemm.CONFIGS[(k, n)] = prev
        per = a.kv_per_clip if name == "m_kv" else PER_STEP[name] * steps
        ok = [v for v in var if var[v]["err"] <= 2.0 * err_c]
        pick = min(ok, key=lambda v: var[v]["us"]) if ok else min(var, key=lambda v: var[v]["us"])
        save = (t_cub - var[pick]["us"]) * per / 1000.0
        r = {"k": k, "n": n, "bias": has_bias, "m": m, "cublas_us": round(t_cub, 2), "err_cublas": err_c,
             "variants": var, "pick": pick, "plain_us": var[pick]["us"], "plain_cfg": var[pick]["cfg"],
             "err_plain": var[pick]["err"], "calls_per_clip": per, "save_ms_per_clip": save,
             "keep": bool(save >= a.min_save_ms and var[pick]["err"] <= 2.0 * err_c)}
        rep["shapes"][name] = r
        print(name, json.dumps({kk: (vv if kk != "variants" else {q: {z: e for z, e in d.items() if z != "top5"}
                                                                  for q, d in vv.items()})
                                for kk, vv in r.items()}), flush=True)
    rep["clip_ms_cublas"] = sum(r["cublas_us"] * r["calls_per_clip"] for r in rep["shapes"].values()) / 1000.0
    rep["clip_ms_kept"] = sum((r["plain_us"] if r["keep"] else r["cublas_us"]) * r["calls_per_clip"]
                              for r in rep["shapes"].values()) / 1000.0
    rep["kept"] = [n for n, r in rep["shapes"].items() if r["keep"]]
    print("per clip: cublas %.2f ms, with kept triton shapes %.2f ms, kept %s" % (
        rep["clip_ms_cublas"], rep["clip_ms_kept"], rep["kept"]), flush=True)
    dump(rep, WORK / ("tune_medium_M%d.json" % a.m))
    return 0


def cmd_post(a):
    return wo.cmd_post(a)


def cmd_crosscheck(a):
    ref = Path(a.ref)
    out = {}
    for k in load_json(RAW / a.arm / "gen.json")["keys"]:
        et, mat = k.split("|")
        rows = []
        for v in range(a.seeds):
            p = ref / ("%s-%s_%02d.npy" % (et, mat, v))
            if not p.is_file():
                continue
            x, y = np.load(RAW / a.arm / p.name), np.load(p)
            rows.append({"v": v, "equal": bool(np.array_equal(x, y)), "shape_equal": x.shape == y.shape,
                         "max_abs": float(np.max(np.abs(x - y))) if x.shape == y.shape else None})
        if rows:
            out[k] = {"equal": sum(r["equal"] for r in rows), "n": len(rows),
                      "max_abs": max(r["max_abs"] or 0.0 for r in rows), "rows": rows}
            print("%-10s %-20s vs %s: %d/%d bit-identical, max abs %.3g" % (
                a.arm, k, ref.name, out[k]["equal"], out[k]["n"], out[k]["max_abs"]), flush=True)
    dump({"arm": a.arm, "ref": str(ref), "keys": out}, WORK / ("crosscheck_%s.json" % a.arm))
    return 0


def tail_cut(base, arm, keys, seeds):
    gb = load_json(RAW / base / "gen.json")["config"]["gen_duration_s"]
    ga = load_json(RAW / arm / "gen.json")["config"]["gen_duration_s"]
    if ga >= gb:
        return None
    per = {}
    for k in keys:
        et, mat = k.split("|")
        ends = [natural_end(np.load(RAW / base / ("%s-%s_%02d.npy" % (et, mat, v)))) for v in range(seeds)]
        per[k] = {"cut": int(sum(e > ga for e in ends)), "n": seeds, "base_end_s_median": med(ends),
                  "base_end_s_max": max(ends)}
    return {"gen_s": ga, "base_gen_s": gb, "per_key": per,
            "pass": bool(all(v["cut"] <= TAIL_MAX for v in per.values()))}


def cmd_compare(a):
    path = WORK / ("compare_%s.json" % a.base)
    out = load_json(path) if path.is_file() else {"base": a.base, "arms": {}}
    bi = load_json(RAW / a.base / "gen.json")
    keys, seeds = bi["keys"], len(bi["seeds"])
    for arm in a.arms:
        r = {"g0": wo.g0_numbers(a.base, arm, keys, seeds), "g1_logmel": wo.g1_numbers(a.base, arm, keys, seeds),
             "static_flags": wo.static_flags(arm), "static_flags_base": wo.static_flags(a.base),
             "tail_cut": tail_cut(a.base, arm, keys, seeds)}
        out["arms"][arm] = r
        print("%-12s vs %-9s G0 raw %.2e fin %.2e equal %d/%d len-mismatch %d | log-mel %.3f dB, seed %.3f dB, ratio "
              "%.3f | static %d (base %d)%s" % (
                  arm, a.base, r["g0"]["raw_max_abs"], r["g0"]["fin_max_abs"], r["g0"]["array_equal"], r["g0"]["n"],
                  r["g0"]["length_mismatch"], r["g1_logmel"]["logmel_db_median"], r["g1_logmel"]["seed_logmel_db"],
                  r["g1_logmel"]["ratio_to_seed"], r["static_flags"], r["static_flags_base"],
                  " | tail cut %s" % json.dumps({k: v["cut"] for k, v in r["tail_cut"]["per_key"].items()})
                  if r["tail_cut"] else ""), flush=True)
        print("             per key ratio: %s" % "  ".join("%s %.2f" % (k, v["ratio"]) for k, v in
                                                        r["g1_logmel"]["per_key"].items()), flush=True)
    dump(out, path)
    return 0


def cmd_gate(a):
    cmp_ = load_json(WORK / ("compare_%s.json" % a.base))["arms"]
    path = WORK / ("gate_%s.json" % a.base)
    out = load_json(path) if path.is_file() else {"base": a.base, "arms": {}, "rep": None, "rep_g0": True}
    if a.rep:
        out.update({"rep": a.rep, "rep_g0": bool(cmp_[a.rep]["g0"]["pass"])})
    rep_ok = bool(out.get("rep_g0", True))
    for arm in a.arms:
        info = load_json(RAW / arm / "gen.json")
        c = cmp_[arm]
        cls = a.cls or info.get("gate_class") or "G1"
        r = {"class": cls, "g0": c["g0"], "g1_logmel": c["g1_logmel"], "tail_cut": c.get("tail_cut"), "reasons": []}
        if a.j4:
            r["j4"] = wo.j4_numbers(a.j4, a.base, arm)
        if a.vocab:
            r["identity"] = wo.vocab_numbers(a.vocab, a.base, arm)
        g0 = bool(c["g0"]["pass"] and rep_ok and not info["stats"].get("graph_error"))
        g1 = []
        if not c["g1_logmel"]["pass"]:
            g1.append("log-mel ratio %.3f above %.2f" % (c["g1_logmel"]["ratio_to_seed"], wo.G1_RATIO))
        g1 += r["j4"]["reasons"] if "j4" in r else ["J4 not run"]
        g1 += r["identity"]["reasons"] if "identity" in r else ["identity floor not run"]
        if c["g0"]["nonfinite_clips"]:
            g1.append("%d clips with NaN or inf" % c["g0"]["nonfinite_clips"])
        if info.get("bad_clips"):
            g1.append("%d clips silent or non-finite" % len(info["bad_clips"]))
        if c["static_flags"] > c["static_flags_base"] + 1:
            g1.append("static flags %d vs base %d" % (c["static_flags"], c["static_flags_base"]))
        if c.get("tail_cut") and not c["tail_cut"]["pass"]:
            g1.append("tail cut %s above %d of %d" % (json.dumps({k: v["cut"] for k, v in
                                                                  c["tail_cut"]["per_key"].items()}),
                                                      TAIL_MAX, len(info["seeds"])))
        if info["stats"].get("graph_error"):
            g1.append("graph or hoist failure: %s" % info["stats"]["graph_error"][:160])
        r["g1_reasons"] = g1
        if cls == "G2":
            r["verdict"] = "G2: listening page, not adopted"
        elif g0:
            r["verdict"] = "PASS G0"
        elif not g1:
            r["verdict"] = "PASS G1"
        else:
            r["verdict"] = "FAIL"
        out["arms"][arm] = r
        print("%-12s %-3s vs %-9s %-32s %s" % (arm, cls, a.base, r["verdict"],
                                               "; ".join(g1) if r["verdict"] != "PASS G0" else ""), flush=True)
    dump(out, path)
    return 0


def h(s):
    return hashlib.md5(s.encode()).hexdigest()


def gate_index():
    gates = {}
    for p in sorted(glob.glob(str(WORK / "gate_*.json"))):
        g = load_json(p)
        for arm, r in g["arms"].items():
            gates.setdefault(arm, {})[g["base"]] = r
    return gates


def cmd_page(a):
    out = OPT / ("page_%s" % a.name)
    if out.is_dir():
        shutil.rmtree(out)
    (out / "a").mkdir(parents=True)
    rp = WORK / ("compare_%s.json" % a.ref)
    ref = load_json(rp)["arms"] if rp.is_file() else {}
    labels, sections, key = {}, [], {}
    for arm in a.arms:
        info = load_json(RAW / arm / "gen.json")
        t = info["timing"]
        lg = ref.get(arm, {}).get("g1_logmel") or {}
        labels[arm] = "%s | %.0f ms per clip | %.2f GB peak | log-mel ratio %s vs %s | %s" % (
            arm, t["cond_hit_median_ms"] or t["median_ms"], info["peak_reserved_gb"],
            "%.2f" % lg["ratio_to_seed"] if lg else "-", a.ref, info["note"])
    letters = "ABCDEFGHIJ"
    for et, mat in GK:
        k = kname(et, mat)
        stem = k.replace("|", "-")
        arms = sorted(a.arms, key=lambda m: h(a.seed + k + m))
        rows = []
        for i, m in enumerate(arms):
            clips = []
            for v in range(G2_SEEDS):
                src = OPT / m / ("%s_%02d.wav" % (stem, v))
                if not src.is_file():
                    continue
                nm = h(a.seed + k + m + str(v))[:10] + ".wav"
                shutil.copy(src, out / "a" / nm)
                clips.append("<audio controls preload='none' src='a/%s'></audio>" % nm)
            key["%s|%s" % (stem, letters[i])] = m
            rows.append("<tr data-k='%s' data-l='%s'><th>%s</th><td class='clips'>%s</td><td>"
                        "<button data-v='best'>best</button><button data-v='worst'>worst</button></td>"
                        "<td class='rev'></td></tr>" % (stem, letters[i], letters[i], "".join(clips)))
        sections.append("<h2>%s</h2><table>%s</table><p><button class='none' data-k='%s'>none sound right</button></p>"
                        % (html.escape(k.replace("|", " ").replace("_", " ")), "".join(rows), stem))
    store = "medium_opt_" + a.name
    page = """<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Gun Speed Candidates</title><style>
:root{--bg:#fafaf8;--fg:#1c1c1a;--mute:#6b6b66;--line:#ddddd8;--acc:#2b5fd9;--bad:#c2410c}
@media (prefers-color-scheme: dark){:root:not([data-theme='light']){--bg:#161615;--fg:#ececea;--mute:#9a9a95;--line:#33332f;--acc:#8fb0ff;--bad:#fb923c}}
:root[data-theme='dark']{--bg:#161615;--fg:#ececea;--mute:#9a9a95;--line:#33332f;--acc:#8fb0ff;--bad:#fb923c}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:24px 16px;max-width:1100px}
table{border-collapse:collapse;width:100%%}td,th{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:middle}
.clips{display:flex;flex-wrap:wrap;gap:6px}audio{width:200px;max-width:100%%;height:32px}
button{font:inherit;padding:3px 10px;margin:0 2px;border:1px solid var(--line);background:transparent;color:var(--fg);border-radius:6px;cursor:pointer}
button.on{border-color:var(--acc);color:var(--acc);font-weight:600}button.on[data-v='worst'],button.none.on{border-color:var(--bad);color:var(--bad)}
p{color:var(--mute);max-width:820px}.rev{color:var(--acc);font-size:13px}h2{margin-top:28px;text-transform:capitalize}
</style></head><body><h1>Gun speed candidates</h1>
<p>Use headphones. Each option is the same prompt and the same four seeds rendered by a different SA3 medium setting,
finished the way the game gets it (natural trim, amplifier, stereo widening, peak level). Letters are shuffled per sound.
Mark the best and the worst option for each sound, or "none sound right". Reveal shows the setting, its speed per clip
and how far it moved from the reference setting (log-mel distance as a fraction of a new seed).</p>
%s
<p><button id='reveal'>Reveal</button> <button id='dl'>Download picks</button> <span id='tally'></span></p>
<script>
const KEY=%s, INFO=%s;
let store={};try{store=JSON.parse(localStorage.getItem('%s')||'{}')}catch(e){}
function save(){try{localStorage.setItem('%s',JSON.stringify(store))}catch(e){}}
function paint(){document.querySelectorAll('tr[data-k]').forEach(tr=>{const k=tr.dataset.k,l=tr.dataset.l,s=store[k]||{};
 tr.querySelectorAll('button').forEach(b=>b.classList.toggle('on',s[b.dataset.v]===l));});
 document.querySelectorAll('button.none').forEach(b=>b.classList.toggle('on',!!(store[b.dataset.k]||{}).none));}
document.querySelectorAll('tr[data-k] button').forEach(b=>b.onclick=()=>{const tr=b.closest('tr'),k=tr.dataset.k;
 store[k]=store[k]||{};store[k][b.dataset.v]=tr.dataset.l;store[k].none=false;save();paint();});
document.querySelectorAll('button.none').forEach(b=>b.onclick=()=>{const k=b.dataset.k;store[k]={none:!(store[k]||{}).none};save();paint();});
document.getElementById('reveal').onclick=()=>{const best={},worst={};
 document.querySelectorAll('tr[data-k]').forEach(tr=>{const m=KEY[tr.dataset.k+'|'+tr.dataset.l];
  tr.querySelector('.rev').textContent=INFO[m];});
 Object.entries(store).forEach(([k,s])=>{if(s.best){const m=KEY[k+'|'+s.best];best[m]=(best[m]||0)+1;}
  if(s.worst){const m=KEY[k+'|'+s.worst];worst[m]=(worst[m]||0)+1;}});
 document.getElementById('tally').textContent='best: '+JSON.stringify(best)+'  worst: '+JSON.stringify(worst);};
document.getElementById('dl').onclick=()=>{const x=document.createElement('a');
 x.href=URL.createObjectURL(new Blob([JSON.stringify({picks:store,key:KEY},null,1)],{type:'application/json'}));
 x.download='medium_opt_picks.json';x.click();};
paint();
</script></body></html>""" % ("".join(sections), json.dumps(key), json.dumps(labels), store, store)
    with open(out / "index.html", "w", encoding="utf-8") as fh:
        fh.write(page)
    print("wrote %s (%d sounds, %d options each)" % (out / "index.html", len(GK), len(a.arms)), flush=True)
    return 0


def cmd_report(a):
    gates = gate_index()
    rows = {}
    for p in sorted(glob.glob(str(RAW / "*" / "gen.json"))):
        info = load_json(p)
        arm = info["arm"]
        t = info["timing"]
        g = gates.get(arm, {})
        cmp_ = {}
        for cp in sorted(glob.glob(str(WORK / "compare_*.json"))):
            c = load_json(cp)
            if arm in c["arms"]:
                cmp_[c["base"]] = {"ratio_to_seed": c["arms"][arm]["g1_logmel"]["ratio_to_seed"],
                                   "g0_raw_max_abs": c["arms"][arm]["g0"]["raw_max_abs"],
                                   "g0_array_equal": c["arms"][arm]["g0"]["array_equal"]}
        rows[arm] = {"config": info["config"], "class": info.get("gate_class"), "vs": info.get("base"),
                     "ms_per_clip": t["median_ms"], "hit_ms": t["cond_hit_median_ms"], "miss_ms": t["cond_miss_median_ms"],
                     "p90_ms": t["p90_ms"], "model_ms": t["model_median_ms"], "finish_ms": t["finish_median_ms"],
                     "stages_ms": t["stages_ms"], "rep_max_abs_diff": t["rep_max_abs_diff"],
                     "load_s": info["load_s"], "warm_s": info["warm_s"],
                     "vram_peak_alloc_gb": info["peak_alloc_gb"], "vram_peak_reserved_gb": info["peak_reserved_gb"],
                     "h1": info.get("h1"), "graph_error": info["stats"].get("graph_error"),
                     "noise_check": info.get("noise_check"), "static_guard": info.get("static_guard"),
                     "compare": cmp_, "gates": g, "verdict": {b: r.get("verdict") for b, r in g.items()},
                     "g2_audio": str(OPT / arm) if info.get("g2_keys") else None, "time": info.get("time")}
    res = {"meta": {"keys": [kname(et, mat) for et, mat in GK], "seeds": 8, "g2_seeds": G2_SEEDS,
                    "finish": FINISH, "g0": "raw and finished max abs diff below %g on all clips, equal lengths"
                    % wo.G0_TOL,
                    "g1": "log-mel ratio to seed <= %.1f on the 0.6 s judge windows, J4 pool R >= base interval low and "
                          "no family Worse (shoot pistol, shoot rifle), J4u artifact paired bootstrap not above 0 "
                          "(reload rifle), identity cons >= base interval low per judged key and one-sided sign test "
                          "p >= 0.05 per judge, no NaN, static flags <= base + 1, tail cut <= %d of 8 per key for "
                          "shorter generation" % (wo.G1_RATIO, TAIL_MAX),
                    "note": a.note, "time": time.strftime("%Y-%m-%d %H:%M:%S")},
           "adopted_defaults": a.adopted, "arms": rows}
    dump(res, OPT / "results.json")
    print("%-12s %-4s %8s %8s %7s %s" % ("arm", "cls", "ms/clip", "hit ms", "VRAM", "verdict"))
    for arm, r in rows.items():
        print("%-12s %-4s %8.1f %8s %7.2f %s" % (arm, r["class"] or "-", r["ms_per_clip"],
                                                "%.1f" % r["hit_ms"] if r["hit_ms"] else "-",
                                                r["vram_peak_reserved_gb"], json.dumps(r["verdict"])))
    print("saved -> %s" % (OPT / "results.json"))
    return 0


def parse(argv):
    ap = argparse.ArgumentParser(description="SA3 medium gun speed candidates: measure, save, gate")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--arm", required=True)
    p.add_argument("--base", choices=["hybrid", "eager"], default="hybrid")
    p.add_argument("--set", nargs="*", default=[])
    p.add_argument("--noise-span", default=None)
    p.add_argument("--hoist-diff", action="store_true")
    p.add_argument("--triton-medium", default=None)
    p.add_argument("--triton-shapes", nargs="*", default=None)
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--stages", action="store_true")
    p.add_argument("--stage-seeds", type=int, default=2)
    p.add_argument("--g2", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--gate", default="G1", choices=["base", "G0", "G1", "G2"])
    p.add_argument("--vs", default=None)
    p.add_argument("--note", default=None)
    p = sub.add_parser("tune")
    p.add_argument("--m", type=int, default=162)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--kv-per-clip", type=int, default=192)
    p.add_argument("--min-save-ms", type=float, default=0.5)
    p.add_argument("--variants", nargs="+", default=["f16", "f32"])
    p = sub.add_parser("post")
    p.add_argument("--arms", nargs="+", required=True)
    p = sub.add_parser("crosscheck")
    p.add_argument("--arm", required=True)
    p.add_argument("--ref", required=True)
    p.add_argument("--seeds", type=int, default=8)
    p = sub.add_parser("compare")
    p.add_argument("--base", required=True)
    p.add_argument("--arms", nargs="+", required=True)
    p = sub.add_parser("gate")
    p.add_argument("--base", required=True)
    p.add_argument("--arms", nargs="+", required=True)
    p.add_argument("--rep", default=None)
    p.add_argument("--j4", default=None)
    p.add_argument("--vocab", default=None)
    p.add_argument("--cls", default=None, choices=["G0", "G1", "G2"])
    p = sub.add_parser("page")
    p.add_argument("--name", required=True)
    p.add_argument("--arms", nargs="+", required=True)
    p.add_argument("--seed", default="mopt1")
    p.add_argument("--ref", required=True)
    p = sub.add_parser("report")
    p.add_argument("--adopted", default="")
    p.add_argument("--note", default="")
    return ap.parse_args(argv)


def main(argv=None):
    a = parse(sys.argv[1:] if argv is None else list(argv))
    return {"run": cmd_run, "tune": cmd_tune, "post": cmd_post, "crosscheck": cmd_crosscheck, "compare": cmd_compare,
            "gate": cmd_gate, "page": cmd_page, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())

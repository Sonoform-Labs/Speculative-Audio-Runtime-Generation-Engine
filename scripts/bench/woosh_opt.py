import argparse
import glob
import hashlib
import html
import json
import math
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

OPT = ROOT / "data" / "_gate" / "opt" / "woosh"
WORK = OPT / "_work"
RAW, FIN, ROUTES, LISTEN = (WORK / d for d in ("raw", "fin", "routes", "listen"))
SERVER = ROOT / "scripts" / "woosh_server.py"
INTENSITY, CONTEXT = "normal", "default"
WK = [("break", "window"), ("footstep", "water"), ("splash", "water"), ("footstep", "gravel"), ("footstep", "stone"),
      ("bullet impact", "glass")]
BAKEOFF_KEYS = [("shoot", "pistol"), ("shoot", "rifle"), ("break", "window"), ("bullet impact", "glass"),
                ("footstep", "gravel"), ("footstep", "stone"), ("footstep", "wood"), ("footstep", "metal"),
                ("footstep", "water"), ("splash", "water"), ("punch", "body"), ("reload", "rifle")]
GUN_FAMILIES = ("blast", "handling")
G2_SEEDS = 4
WARM_PROMPT = "warmup, a soft thud"
G0_TOL = 1e-4
G1_RATIO = 0.5
J4_POOL_KEYS = ["break|window", "bullet_impact|glass", "footstep|gravel", "footstep|stone", "splash|water"]
J4_ART_KEYS = ["footstep|water"]
VOCAB_JUDGES = ("ast", "panns", "clap")
BOOT, BOOT_RNG = 2000, 20261014


def ev(et):
    return et.replace(" ", "_")


def kname(et, mat):
    return "%s|%s" % (ev(et), mat)


def fname(et, mat, v):
    return "%s-%s_%02d" % (ev(et), mat, v)


def dump(obj, p):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


def load_json(p):
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def g2_keys():
    from sarge.amp import family_for
    return [(et, mat) for et, mat in BAKEOFF_KEYS if family_for(et) not in GUN_FAMILIES]


def pct(xs, q):
    return float(np.percentile(np.asarray(xs, dtype=float), q)) if len(xs) else None


def med(xs):
    return float(np.median(np.asarray(xs, dtype=float))) if len(xs) else None


class Stages:
    def __init__(self, torch):
        self.torch, self.rows, self.cur, self.t = torch, [], None, 0.0

    def start(self):
        self.torch.cuda.synchronize()
        self.cur, self.t = {}, time.perf_counter()

    def __call__(self, name):
        self.torch.cuda.synchronize()
        now = time.perf_counter()
        self.cur[name] = self.cur.get(name, 0.0) + (now - self.t) * 1000.0
        self.t = now

    def end(self):
        self.rows.append(self.cur)


def istft_check(ws, gen):
    torch = gen.torch
    head = gen.ldm.autoencoder.autoencoder.decoder.head
    g = torch.Generator(device="cuda").manual_seed(1)
    spec = torch.complex(torch.randn(1, head.n_fft // 2 + 1, gen.frames, device="cuda", generator=g),
                         torch.randn(1, head.n_fft // 2 + 1, gen.frames, device="cuda", generator=g))
    with torch.inference_mode():
        a = head.istft(spec)
        b = ws.Istft(head.istft, gen.frames)(spec)
    return {"equal": bool(torch.equal(a, b)), "max_abs": float((a - b).abs().max())}


def cmd_run(a):
    sys.path.insert(0, str(Path(a.server).resolve().parent))
    import woosh_server as ws
    import torch
    from sarge.prompts import build
    from sarge.types import AudioClip, CacheKey, seed_for
    for d in (RAW / a.arm, FIN / a.arm):
        if d.is_dir():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    t0 = time.perf_counter()
    gen = ws.make_gen(a.model, a.frames, a.steps, a.solver, cond_cache=a.cond_cache, graph=a.graph,
                      precision=a.precision, ae_precision=a.ae_precision, batch_cfg=a.batch_cfg,
                      time_dtype=a.time_dtype, **({"mod_cache": True} if a.mod_cache else {}))
    torch.cuda.synchronize()
    load_s = time.perf_counter() - t0
    load_peak = {"peak_alloc_gb": torch.cuda.max_memory_allocated() / 2 ** 30,
                 "peak_reserved_gb": torch.cuda.max_memory_reserved() / 2 ** 30}
    torch.cuda.reset_peak_memory_stats()
    print("loaded woosh %s in %.1f s, peak %s" % (a.model, load_s, json.dumps(load_peak)), flush=True)
    t1 = time.perf_counter()
    gen(WARM_PROMPT, 0)
    torch.cuda.synchronize()
    warm_s = time.perf_counter() - t1
    print("warm-up %.2f s, graph error %s" % (warm_s, gen.stats["graph_error"]), flush=True)
    keys = list(WK)
    extra = [k for k in g2_keys() if k not in keys] if a.g2 else []
    prompts, saved, calls, finish_ms, bad = {}, {}, [], [], []
    nfe0 = len(gen.stats["nfe"])
    for et, mat in keys + extra:
        key = CacheKey(et, mat, INTENSITY, CONTEXT)
        p = build(et, mat, INTENSITY, CONTEXT)
        prompts[kname(et, mat)] = p
        nseeds = a.seeds if (et, mat) in keys else G2_SEEDS
        ks = []
        for v in range(nseeds):
            miss0 = gen.cache.misses
            torch.cuda.synchronize()
            t = time.perf_counter()
            x = gen(p, seed_for(key, v))
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t) * 1000.0
            x = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
            if x.ndim == 1:
                x = x[:, None]
            t2 = time.perf_counter()
            if np.all(np.isfinite(x)) and np.any(np.abs(x) > 1e-6):
                y = ws.finish(x, et, INTENSITY, True, True)
            else:
                bad.append(fname(et, mat, v))
                y = np.zeros((x.shape[0], 2), dtype=np.float32)
            finish_ms.append((time.perf_counter() - t2) * 1000.0)
            if (et, mat) in keys:
                np.save(RAW / a.arm / (fname(et, mat, v) + ".npy"), x)
                np.save(FIN / a.arm / (fname(et, mat, v) + ".npy"), y)
                saved[fname(et, mat, v)] = x
                calls.append({"pass": 0, "key": kname(et, mat), "v": v, "ms": ms,
                              "miss": bool(gen.cache.misses > miss0 or not gen.cache.size)})
            if a.g2 and v < G2_SEEDS:
                (OPT / a.arm).mkdir(parents=True, exist_ok=True)
                AudioClip(y, ws.SR).to_wav(str(OPT / a.arm / (fname(et, mat, v) + ".wav")))
            ks.append(ms)
        print("%-10s %-22s median %.1f ms (%d seeds)" % (a.arm, kname(et, mat), med(ks), nseeds), flush=True)
    nfe_save = gen.stats["nfe"][nfe0:]
    rep_diff = []
    for rep in range(a.reps):
        for et, mat in keys:
            key = CacheKey(et, mat, INTENSITY, CONTEXT)
            p = prompts[kname(et, mat)]
            for v in range(a.seeds):
                miss0 = gen.cache.misses
                torch.cuda.synchronize()
                t = time.perf_counter()
                x = gen(p, seed_for(key, v))
                torch.cuda.synchronize()
                ms = (time.perf_counter() - t) * 1000.0
                calls.append({"pass": rep + 1, "key": kname(et, mat), "v": v, "ms": ms,
                              "miss": bool(gen.cache.misses > miss0 or not gen.cache.size)})
                rep_diff.append(float(np.max(np.abs(np.asarray(x, dtype=np.float32).reshape(-1) -
                                                    saved[fname(et, mat, v)].reshape(-1)))))
    stages = None
    if a.stages:
        sg = Stages(torch)
        for et, mat in keys:
            key = CacheKey(et, mat, INTENSITY, CONTEXT)
            for v in range(a.stage_seeds):
                sg.start()
                gen(prompts[kname(et, mat)], seed_for(key, v), mark=sg)
                sg.end()
        names = sorted({n for r in sg.rows for n in r})
        stages = {n: med([r.get(n, 0.0) for r in sg.rows]) for n in names}
        stages["total"] = med([sum(r.values()) for r in sg.rows])
        print("stages (median ms): %s" % json.dumps({k: round(v, 2) for k, v in stages.items()}), flush=True)
    peak = {"peak_alloc_gb": torch.cuda.max_memory_allocated() / 2 ** 30,
            "peak_reserved_gb": torch.cuda.max_memory_reserved() / 2 ** 30}
    all_ms = [c["ms"] for c in calls]
    miss_ms = [c["ms"] for c in calls if c["miss"]]
    hit_ms = [c["ms"] for c in calls if not c["miss"]]
    nfe_wk = nfe_save[:len(keys) * a.seeds]
    timing = {"median_ms": med(all_ms), "p90_ms": pct(all_ms, 90), "p10_ms": pct(all_ms, 10), "n": len(all_ms),
              "save_pass_median_ms": med([c["ms"] for c in calls if c["pass"] == 0]),
              "cond_miss_median_ms": med(miss_ms), "cond_hit_median_ms": med(hit_ms), "n_miss": len(miss_ms),
              "n_hit": len(hit_ms), "finish_median_ms": med(finish_ms), "stages_ms": stages,
              "nfe": {"median": med(nfe_wk), "min": int(min(nfe_wk)), "max": int(max(nfe_wk)),
                      "dit_forwards_median": med(nfe_wk) * (2 if a.model == "flow" else 1)} if nfe_wk else None,
              "rep_max_abs_diff": max(rep_diff) if rep_diff else None}
    cfg = {"model": a.model, "frames": a.frames, "seconds": (a.frames - 1) * 480 / ws.SR, "steps": a.steps,
           "solver": a.solver if a.model == "flow" else "euler", "cond_cache": a.cond_cache, "graph": a.graph,
           "precision": a.precision, "ae_precision": a.ae_precision, "batch_cfg": a.batch_cfg,
           "time_dtype": a.time_dtype, "mod_cache": bool(a.mod_cache),
           "renoise": ws.renoise_for(a.steps) if a.model == "dflow" else None}
    info = {"arm": a.arm, "family": "woosh", "model": "woosh-" + a.model, "gate_class": a.gate, "base": a.base,
            "note": a.note or "woosh %s %s" % (a.model, json.dumps(cfg)), "config": cfg, "sr": ws.SR, "channels": 1,
            "seeds": list(range(a.seeds)), "keys": [kname(et, mat) for et, mat in keys], "prompts": prompts,
            "g2_keys": [kname(et, mat) for et, mat in keys + extra] if a.g2 else [],
            "g2_seeds": G2_SEEDS if a.g2 else 0,
            "intensity": INTENSITY, "context": CONTEXT, "load_s": load_s, "warm_s": warm_s,
            "median_s": timing["median_ms"] / 1000.0, "load_peak": load_peak, **peak, "timing": timing,
            "calls": calls, "cond": {"size": gen.cache.size, "hits": gen.cache.hits, "misses": gen.cache.misses},
            "bad_clips": bad,
            "stats": {k: v for k, v in gen.stats.items() if k != "nfe"},
            "istft_check": istft_check(ws, gen) if a.graph == "full" else None,
            "torch": torch.__version__, "device": torch.cuda.get_device_name(0),
            "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    dump(info, RAW / a.arm / "gen.json")
    print("%-10s done: median %.1f ms p90 %.1f ms (miss %.1f / hit %.1f), warm-up %.1f s, peak %.2f GB alloc %.2f GB "
          "reserved, nfe %s, rep diff %s, graph %s" % (
              a.arm, timing["median_ms"], timing["p90_ms"], timing["cond_miss_median_ms"] or float("nan"),
              timing["cond_hit_median_ms"] or float("nan"), warm_s, peak["peak_alloc_gb"], peak["peak_reserved_gb"],
              json.dumps(timing["nfe"]), timing["rep_max_abs_diff"], gen.stats["graph_error"] or "ok"), flush=True)
    if bad:
        print("%-10s %d clips non-finite or silent: %s" % (a.arm, len(bad), " ".join(bad[:8])), flush=True)
    return 0


def bakeoff_mod():
    import bakeoff
    bakeoff.RAW, bakeoff.LISTEN, bakeoff.ROUTES = str(RAW), str(LISTEN), str(ROUTES)
    return bakeoff


def cmd_post(a):
    bk = bakeoff_mod()
    return bk.cmd_post(argparse.Namespace(arms=a.arms))


def read_wav(p):
    from sarge.types import AudioClip
    return np.asarray(AudioClip.from_wav(str(p)).samples, dtype=np.float64)


def static_flags(arm):
    r = load_json(ROUTES / arm / "route.json")
    return int(sum(sum(v["flags"]) for v in r["static_rule"].values()))


def g0_numbers(base, arm, keys, seeds):
    raw_d, fin_d, eq, nonfin, len_bad = [], [], 0, 0, 0
    for k in keys:
        et, mat = k.split("|")
        for v in range(seeds):
            n = "%s-%s_%02d.npy" % (et, mat, v)
            a, b = np.load(RAW / base / n), np.load(RAW / arm / n)
            nonfin += int(not np.all(np.isfinite(b)))
            if a.shape != b.shape:
                len_bad += 1
                raw_d.append(float("inf"))
            else:
                raw_d.append(float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))))
                eq += int(np.array_equal(a, b))
            fa, fb = np.load(FIN / base / n), np.load(FIN / arm / n)
            if fa.shape != fb.shape:
                len_bad += 1
                fin_d.append(float("inf"))
            else:
                fin_d.append(float(np.max(np.abs(fa.astype(np.float64) - fb.astype(np.float64)))))
    return {"raw_max_abs": max(raw_d), "raw_max_abs_median": med(raw_d), "fin_max_abs": max(fin_d),
            "fin_max_abs_median": med(fin_d), "array_equal": eq, "n": len(raw_d), "length_mismatch": len_bad,
            "nonfinite_clips": nonfin, "pass": bool(max(raw_d) < G0_TOL and max(fin_d) < G0_TOL and not len_bad)}


def g1_numbers(base, arm, keys, seeds):
    from compile_probe import logmel
    lm = {}

    def L(r, k, v):
        et, mat = k.split("|")
        kk = (r, k, v)
        if kk not in lm:
            lm[kk] = logmel(read_wav(ROUTES / r / ("%s-%s_%02d.wav" % (et, mat, v))))
        return lm[kk]

    seed_d, dist, per_key = [], [], {}
    for k in keys:
        sk = [float(np.mean(np.abs(L(base, k, v) - L(base, k, v + 1)))) for v in range(seeds - 1)]
        dk = [float(np.mean(np.abs(L(base, k, v) - L(arm, k, v)))) for v in range(seeds)]
        seed_d += sk
        dist += dk
        per_key[k] = {"dist_median": med(dk), "seed_median": med(sk), "ratio": med(dk) / max(med(sk), 1e-12)}
    sm = med(seed_d)
    return {"logmel_db_median": med(dist), "seed_logmel_db": sm, "ratio_to_seed": med(dist) / max(sm, 1e-12),
            "per_key": per_key, "pass": bool(med(dist) / max(sm, 1e-12) <= G1_RATIO)}


def cmd_compare(a):
    path = WORK / ("compare_%s.json" % a.base)
    out = load_json(path) if path.is_file() else {"base": a.base, "arms": {}}
    bi = load_json(RAW / a.base / "gen.json")
    keys, seeds = bi["keys"], len(bi["seeds"])
    for arm in a.arms:
        r = {"g0": g0_numbers(a.base, arm, keys, seeds), "g1_logmel": g1_numbers(a.base, arm, keys, seeds),
             "static_flags": static_flags(arm), "static_flags_base": static_flags(a.base)}
        out["arms"][arm] = r
        print("%-12s vs %-8s G0 raw %.2e fin %.2e equal %d/%d len-mismatch %d | log-mel %.3f dB, seed %.3f dB, ratio "
              "%.3f | static %d (base %d)" % (
                  arm, a.base, r["g0"]["raw_max_abs"], r["g0"]["fin_max_abs"], r["g0"]["array_equal"], r["g0"]["n"],
                  r["g0"]["length_mismatch"], r["g1_logmel"]["logmel_db_median"], r["g1_logmel"]["seed_logmel_db"],
                  r["g1_logmel"]["ratio_to_seed"], r["static_flags"], r["static_flags_base"]), flush=True)
        print("             per key ratio: %s" % "  ".join("%s %.2f" % (k, v["ratio"]) for k, v in
                                                        r["g1_logmel"]["per_key"].items()), flush=True)
    dump(out, path)
    return 0


def j4_numbers(j4dir, base, arm):
    res = load_json(Path(j4dir) / "results.json")
    cells = res.get("cells", {})
    out = {"pool": {}, "art": {}, "family_calls": {}, "pass": True, "reasons": []}
    for k in J4_POOL_KEYS:
        cb, ca = cells.get("%s|%s" % (base, k), {}), cells.get("%s|%s" % (arm, k), {})
        rb, rbc = (cb.get("R_J") or {}).get("J4"), (cb.get("R_J_ci") or {}).get("J4")
        ra = (ca.get("R_J") or {}).get("J4")
        if ra is None or rbc is None:
            out["pool"][k] = {"note": "no J4 cell"}
            continue
        ok = bool(ra >= rbc[0])
        out["pool"][k] = {"R_base": rb, "R_base_ci": rbc, "R_arm": ra, "pass": ok}
        if not ok:
            out["pass"] = False
            out["reasons"].append("J4 %s R %.3f below base interval %.3f" % (k, ra, rbc[0]))
    exp = ((res.get("experiment") or {}).get("routes") or {}).get(arm, {})
    for fam, v in (exp.get("families") or {}).items():
        out["family_calls"][fam] = v.get("call")
        if v.get("call") == "Worse":
            out["pass"] = False
            out["reasons"].append("J4 family %s called Worse" % fam)
    import csv
    lat = {}
    with open(Path(j4dir) / "clips.csv", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["key"] in J4_ART_KEYS and row["kind"] == "gen" and row["route"] in (base, arm):
                try:
                    lat[(row["key"], row["route"], int(row["seed"]))] = max(float(row["lat_J4u"]), 0.0)
                except (KeyError, ValueError):
                    pass
    rng = np.random.default_rng(BOOT_RNG)
    for k in J4_ART_KEYS:
        seeds = sorted({s for kk, r, s in lat if kk == k and r == base} &
                       {s for kk, r, s in lat if kk == k and r == arm})
        if not seeds:
            out["art"][k] = {"note": "no J4u rows"}
            continue
        d = np.array([lat[(k, arm, s)] - lat[(k, base, s)] for s in seeds])
        bs = d[rng.integers(0, len(d), size=(BOOT, len(d)))].mean(1)
        lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
        ok = not lo > 0
        out["art"][k] = {"delta_mean": float(d.mean()), "ci": [lo, hi], "n": len(seeds), "pass": ok}
        if not ok:
            out["pass"] = False
            out["reasons"].append("J4u %s artifact delta interval [%.3f, %.3f] above 0" % (k, lo, hi))
    return out


def sign_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return float(sum(math.comb(n, i) for i in range(b, n + 1)) / 2 ** n)


def vocab_numbers(vjson, base, arm):
    V = load_json(vjson)
    cons, gen = V["consensus"], V["generated"]
    out = {"keys": {}, "sign": {}, "pass": True, "reasons": []}
    pairs = {j: [0, 0] for j in VOCAB_JUDGES}
    for k in [kname(et, mat) for et, mat in WK]:
        cb, ca = cons.get("gen:%s|%s" % (base, k), {}), cons.get("gen:%s|%s" % (arm, k), {})
        if cb.get("cons") is None or ca.get("cons") is None:
            out["keys"][k] = {"note": cb.get("reason") or ca.get("reason") or "not judged"}
            continue
        ok = bool(ca["cons"] >= cb["cons_ci"][0])
        out["keys"][k] = {"cons_base": cb["cons"], "cons_base_ci": cb["cons_ci"], "cons_arm": ca["cons"], "pass": ok}
        if not ok:
            out["pass"] = False
            out["reasons"].append("identity %s cons %.2f below base interval %.2f" % (k, ca["cons"], cb["cons_ci"][0]))
        for j, jv in (cb.get("judges") or {}).items():
            if j not in pairs:
                continue
            var = jv["variant"]
            hb = gen.get("gen:%s|%s" % (base, k), {}).get(var, {}).get("hits")
            ha = gen.get("gen:%s|%s" % (arm, k), {}).get(var, {}).get("hits")
            if hb is None or ha is None:
                continue
            for x, y in zip(hb, ha):
                if x and not y:
                    pairs[j][0] += 1
                elif y and not x:
                    pairs[j][1] += 1
    for j, (b, c) in pairs.items():
        p = sign_p(b, c)
        out["sign"][j] = {"base_only": b, "arm_only": c, "p_worse": p, "pass": bool(p >= 0.05)}
        if p < 0.05:
            out["pass"] = False
            out["reasons"].append("identity sign test %s: %d lost vs %d gained, p %.3f" % (j, b, c, p))
    return out


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
        cls = info.get("gate_class") or "G1"
        r = {"class": cls, "g0": c["g0"], "g1_logmel": c["g1_logmel"], "reasons": []}
        if a.j4:
            r["j4"] = j4_numbers(a.j4, a.base, arm)
        if a.vocab:
            r["identity"] = vocab_numbers(a.vocab, a.base, arm)
        g0 = bool(c["g0"]["pass"] and rep_ok and not info["stats"].get("graph_error"))
        g1 = []
        if not c["g1_logmel"]["pass"]:
            g1.append("log-mel ratio %.3f above %.2f" % (c["g1_logmel"]["ratio_to_seed"], G1_RATIO))
        if "j4" in r:
            g1 += r["j4"]["reasons"]
        else:
            g1.append("J4 not run")
        if "identity" in r:
            g1 += r["identity"]["reasons"]
        else:
            g1.append("identity floor not run")
        if c["g0"]["nonfinite_clips"]:
            g1.append("%d clips with NaN or inf" % c["g0"]["nonfinite_clips"])
        if c["static_flags"] > c["static_flags_base"] + 1:
            g1.append("static flags %d vs base %d" % (c["static_flags"], c["static_flags_base"]))
        if info["stats"].get("graph_error"):
            g1.append("graph capture failed: %s" % info["stats"]["graph_error"][:120])
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
        print("%-12s %-3s %-34s %s" % (arm, cls, r["verdict"], "; ".join(g1) if r["verdict"] != "PASS G0" else ""),
              flush=True)
    dump(out, path)
    return 0


def h(s):
    return hashlib.md5(s.encode()).hexdigest()


def cmd_page(a):
    out = OPT / ("page_%s" % a.name)
    if out.is_dir():
        shutil.rmtree(out)
    (out / "a").mkdir(parents=True)
    labels, sections, key = {}, [], {}
    gates = {}
    for p in sorted(glob.glob(str(WORK / "gate_*.json"))):
        g = load_json(p)
        for arm, r in g["arms"].items():
            gates[arm] = dict(r, base=g["base"])
    rp = WORK / ("compare_%s.json" % a.ref)
    ref = load_json(rp)["arms"] if rp.is_file() else {}
    for arm in a.arms:
        info = load_json(RAW / arm / "gen.json")
        g = gates.get(arm, {})
        t = info["timing"]
        lg = ref.get(arm, {}).get("g1_logmel") or {}
        fast = RAW / (arm + "_a") / "gen.json"
        extra = " (%.0f ms at the adopted precision)" % load_json(fast)["timing"]["median_ms"] if fast.is_file() else ""
        labels[arm] = "%s | %.0f ms per clip%s | %.2f GB peak | log-mel ratio %s vs %s | %s" % (
            arm, t["median_ms"], extra, info["peak_reserved_gb"],
            "%.2f" % lg["ratio_to_seed"] if lg else "-", a.ref, info["note"])
    letters = "ABCDEFGHIJ"
    keys = [kname(et, mat) for et, mat in g2_keys()]
    for k in keys:
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
    store = "woosh_opt_" + a.name
    page = """<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Woosh Speed Candidates</title><style>
:root{--bg:#fafaf8;--fg:#1c1c1a;--mute:#6b6b66;--line:#ddddd8;--acc:#2b5fd9;--bad:#c2410c}
@media (prefers-color-scheme: dark){:root:not([data-theme='light']){--bg:#161615;--fg:#ececea;--mute:#9a9a95;--line:#33332f;--acc:#8fb0ff;--bad:#fb923c}}
:root[data-theme='dark']{--bg:#161615;--fg:#ececea;--mute:#9a9a95;--line:#33332f;--acc:#8fb0ff;--bad:#fb923c}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:24px 16px;max-width:1100px}
table{border-collapse:collapse;width:100%%}td,th{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:middle}
.clips{display:flex;flex-wrap:wrap;gap:6px}audio{width:200px;max-width:100%%;height:32px}
button{font:inherit;padding:3px 10px;margin:0 2px;border:1px solid var(--line);background:transparent;color:var(--fg);border-radius:6px;cursor:pointer}
button.on{border-color:var(--acc);color:var(--acc);font-weight:600}button.on[data-v='worst'],button.none.on{border-color:var(--bad);color:var(--bad)}
p{color:var(--mute);max-width:820px}.rev{color:var(--acc);font-size:13px}h2{margin-top:28px;text-transform:capitalize}
</style></head><body><h1>Woosh speed candidates</h1>
<p>Use headphones. Each option is the same prompt and the same four seeds rendered by a different Woosh setting, finished
the way the game gets it (natural trim, amplifier, stereo widening, peak level). Letters are shuffled per sound. Mark the
best and the worst option for each sound, or "none sound right". Reveal shows the setting, its speed per clip and how far
it moved from the base setting (log-mel distance as a fraction of a new seed).</p>
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
 x.download='woosh_opt_picks.json';x.click();};
paint();
</script></body></html>""" % ("".join(sections), json.dumps(key), json.dumps(labels), store, store)
    with open(out / "index.html", "w", encoding="utf-8") as fh:
        fh.write(page)
    print("wrote %s (%d sounds, %d options each)" % (out / "index.html", len(keys), len(a.arms)), flush=True)
    return 0


def cmd_report(a):
    gates, reps = {}, {}
    for p in sorted(glob.glob(str(WORK / "gate_*.json"))):
        g = load_json(p)
        if g.get("rep"):
            reps[g["rep"]] = g.get("rep_g0")
        for arm, r in g["arms"].items():
            gates[arm] = dict(r, base=g["base"])
    rows = {}
    for p in sorted(glob.glob(str(RAW / "*" / "gen.json"))):
        info = load_json(p)
        arm = info["arm"]
        t = info["timing"]
        g = dict(gates.get(arm, {}))
        if not g.get("verdict"):
            if info.get("bad_clips"):
                g["verdict"] = "FAIL: %d of %d clips non-finite or silent" % (
                    len(info["bad_clips"]), len(info["keys"]) * len(info["seeds"]) + 3 * G2_SEEDS * bool(info["g2_keys"]))
            elif info.get("gate_class") == "base":
                g["verdict"] = "baseline"
            elif arm in reps:
                g["verdict"] = "determinism control, G0 %s" % ("pass" if reps[arm] else "fail")
        rows[arm] = {"config": info["config"], "class": info.get("gate_class"),
                     "base": g.get("base") or info.get("base"),
                     "ms_per_clip": t["median_ms"], "p90_ms": t["p90_ms"], "cond_miss_ms": t["cond_miss_median_ms"],
                     "cond_hit_ms": t["cond_hit_median_ms"], "finish_ms": t["finish_median_ms"],
                     "stages_ms": t["stages_ms"], "nfe": t["nfe"], "rep_max_abs_diff": t["rep_max_abs_diff"],
                     "load_s": info["load_s"], "warm_s": info["warm_s"],
                     "vram_peak_alloc_gb": info["peak_alloc_gb"], "vram_peak_reserved_gb": info["peak_reserved_gb"],
                     "graph_error": info["stats"].get("graph_error"), "istft_check": info.get("istft_check"),
                     "g0": g.get("g0"), "g1_logmel": g.get("g1_logmel"), "j4": g.get("j4"),
                     "identity": g.get("identity"), "g1_reasons": g.get("g1_reasons"), "verdict": g.get("verdict"),
                     "g2_audio": str(OPT / arm) if info.get("g2_keys") else None, "time": info.get("time")}
    res = {"meta": {"keys_g0_g1": [kname(et, mat) for et, mat in WK], "seeds_g0_g1": 8,
                    "g2_keys": [kname(et, mat) for et, mat in g2_keys()], "g2_seeds": G2_SEEDS,
                    "g0": "raw and finished max abs diff below %g on all clips, equal lengths" % G0_TOL,
                    "g1": "log-mel ratio to seed <= %.1f on the 0.6 s judge windows, J4 pool R >= base interval low "
                          "and "
                          "no family Worse, J4u artifact paired bootstrap not above 0, identity cons >= base interval "
                          "low per key and one-sided sign test p >= 0.05 per judge, no NaN, static flags <= base + 1"
                          % G1_RATIO,
                    "note": a.note, "time": time.strftime("%Y-%m-%d %H:%M:%S")},
           "adopted_defaults": a.adopted, "arms": rows}
    dump(res, OPT / "results.json")
    print("%-12s %-4s %8s %8s %7s %-28s" % ("arm", "cls", "ms/clip", "hit ms", "VRAM", "verdict"))
    for arm, r in rows.items():
        print("%-12s %-4s %8.1f %8s %7.2f %-28s" % (arm, r["class"] or "-", r["ms_per_clip"],
                                                   "%.1f" % r["cond_hit_ms"] if r["cond_hit_ms"] else "-",
                                                   r["vram_peak_reserved_gb"], r["verdict"] or "-"))
    print("saved -> %s" % (OPT / "results.json"))
    return 0


def parse(argv):
    ap = argparse.ArgumentParser(description="Woosh speed candidates: measure, save, gate")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--arm", required=True)
    p.add_argument("--model", choices=["dflow", "flow"], required=True)
    p.add_argument("--frames", type=int, default=101)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--solver", default="dopri5")
    p.add_argument("--cond-cache", type=int, default=0)
    p.add_argument("--graph", default="none")
    p.add_argument("--precision", default="fp32")
    p.add_argument("--ae-precision", default="fp32")
    p.add_argument("--batch-cfg", action="store_true")
    p.add_argument("--time-dtype", default="float64")
    p.add_argument("--mod-cache", action="store_true")
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--stages", action="store_true")
    p.add_argument("--stage-seeds", type=int, default=2)
    p.add_argument("--g2", action="store_true")
    p.add_argument("--gate", default="G1", choices=["base", "G0", "G1", "G2"])
    p.add_argument("--base", default=None)
    p.add_argument("--note", default=None)
    p.add_argument("--server", default=str(SERVER))
    p = sub.add_parser("post")
    p.add_argument("--arms", nargs="+", required=True)
    p = sub.add_parser("compare")
    p.add_argument("--base", required=True)
    p.add_argument("--arms", nargs="+", required=True)
    p = sub.add_parser("gate")
    p.add_argument("--base", required=True)
    p.add_argument("--arms", nargs="+", required=True)
    p.add_argument("--rep", default=None)
    p.add_argument("--j4", default=None)
    p.add_argument("--vocab", default=None)
    p = sub.add_parser("page")
    p.add_argument("--name", required=True)
    p.add_argument("--arms", nargs="+", required=True)
    p.add_argument("--seed", default="wopt1")
    p.add_argument("--ref", required=True)
    p = sub.add_parser("report")
    p.add_argument("--adopted", default="")
    p.add_argument("--note", default="")
    return ap.parse_args(argv)


def main(argv=None):
    a = parse(sys.argv[1:] if argv is None else list(argv))
    return {"run": cmd_run, "post": cmd_post, "compare": cmd_compare, "gate": cmd_gate, "page": cmd_page,
            "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if not os.path.isfile(os.path.join(HERE, "_bootstrap.py")):
    HERE = r"D:\Python\SAGE\scripts\bench"
SCRIPTS = os.path.dirname(HERE)
for _p in (SCRIPTS, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _bootstrap

import argparse
import glob
import json
import math
import shutil
import time
from types import SimpleNamespace

import numpy as np

from sarge.generators.stable_audio import _extract
from sarge.loudness import normalize_loudness, target_for
from sarge.types import AudioClip, CacheKey

ROOT = os.path.dirname(SCRIPTS)
OUT = os.path.join(ROOT, "data", "_gate", "prompt_search")
CANDS = os.path.join(OUT, "candidates.json")
INTENSITY = "normal"
WIN_S = 0.6
SPACE_KEYS = ["shoot|pistol", "shoot|rifle", "footstep|stone", "break|window"]
SPACES = ["indoor_small", "indoor_large", "tunnel"]
STYLES = ["library", "descriptive"]
WATER_KEYS = ["footstep|water", "splash|water"]
NEAREST = {
    "bullet impact|glass": ("glass_break", None),
    "bullet impact|metal": (None, None),
    "bullet impact|wood": (None, None),
    "footstep|glass": ("footstep", None),
    "footstep|water": ("splash", None),
    "splash|water": ("splash", None),
    "land|stone": ("footstep", "stone"),
    "equip|rifle": ("gun_mech", None),
}


def fname(key, v):
    et, mat = key.split("|")
    return "%s-%s_%02d.wav" % (et.replace(" ", "_"), mat, v)


def load_json(p):
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def dump_json(obj, p):
    os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, ensure_ascii=False,
                  default=lambda o: o.tolist() if hasattr(o, "tolist") else list(o) if isinstance(o, tuple) else str(o))


def window_clip(a, sr, intensity=INTENSITY):
    a = np.asarray(a, dtype=np.float32)
    while a.ndim > 2:
        a = a[0]
    mono = a.mean(axis=0) if a.ndim == 2 else a
    n = int(WIN_S * sr)
    if mono.shape[0] < n:
        mono = np.pad(mono, (0, n - mono.shape[0]))
    return normalize_loudness(_extract(mono, sr, WIN_S), target_rms=target_for(intensity)).astype(np.float32)


class Tap:
    def __init__(self, g):
        self.orig = g._finish
        self.raw = {}
        self.fallbacks = 0
        g._finish = self

    def __call__(self, a, key):
        out = self.orig(a, key)
        self.raw[id(out)] = (out, np.array(a, dtype=np.float32, copy=True))
        return out

    def window(self, samples, sr):
        hit = self.raw.get(id(samples))
        self.raw.clear()
        if hit is not None and hit[0] is samples:
            return window_clip(hit[1], sr), False
        self.fallbacks += 1
        return window_clip(samples, sr), True


def build_generator():
    import run_sidecar
    from sarge.generators.stable_audio3 import StableAudio3Generator
    kw = dict(run_sidecar.TIERS["default"], fast_load=True)
    return StableAudio3Generator(**kw), kw


def sync():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def render(g, tap, key, prompt, variants, nat_dir, win_dir=None):
    et, mat = key.split("|")
    ck = CacheKey(et, mat, INTENSITY)
    g.prompts[str(ck)] = prompt
    if g._prompt(ck) != prompt:
        raise SystemExit("prompt override ignored for %s" % ck)
    ms, fb = [], 0
    for v in variants:
        sync()
        t1 = time.perf_counter()
        clip = g.generate(ck, variant=v)
        sync()
        ms.append((time.perf_counter() - t1) * 1000.0)
        win, used_fb = tap.window(clip.samples, g.sample_rate)
        fb += int(used_fb)
        clip.to_wav(os.path.join(nat_dir, fname(key, v)))
        if win_dir is not None:
            if win.shape[0] != int(WIN_S * g.sample_rate):
                raise SystemExit("window clip for %s v%d has %d samples" % (key, v, win.shape[0]))
            AudioClip(win, g.sample_rate).to_wav(os.path.join(win_dir, fname(key, v)))
    return ms, fb


def cmd_gen(args):
    C = load_json(args.candidates)
    keys = args.keys or C["keys"]
    slots = args.slots or C["slots"]
    t0 = time.perf_counter()
    g, kw = build_generator()
    g.ensure_ready()
    load_s = time.perf_counter() - t0
    tap = Tap(g)
    t1 = time.perf_counter()
    g.generate(CacheKey("shoot", "rifle", INTENSITY), variant=99)
    g.generate(CacheKey("splash", "water", INTENSITY), variant=99)
    tap.raw.clear()
    warm_s = time.perf_counter() - t1
    print("generator ready %.1f s, warm-up %.1f s, sr %d" % (load_s, warm_s, g.sample_rate), flush=True)
    timing = {"load_s": load_s, "warm_s": warm_s, "slots": {}}
    for slot in slots:
        rd = os.path.join(args.out_dir, "routes", slot)
        nd = os.path.join(args.out_dir, "natural", slot)
        os.makedirs(rd, exist_ok=True)
        os.makedirs(nd, exist_ok=True)
        st0 = dict(g.static_stats)
        ts = time.perf_counter()
        prompts, per_key, all_ms, fbs = {}, {}, [], 0
        for key in keys:
            p = C["candidates"][key][slot]
            prompts[key] = p
            ms, fb = render(g, tap, key, p, range(args.seeds), nd, rd)
            per_key[key] = float(np.median(ms))
            all_ms += ms
            fbs += fb
        static = {a: int(g.static_stats.get(a, 0)) - int(st0.get(a, 0)) for a in g.static_stats}
        secs = time.perf_counter() - ts
        info = {"steps": int(kw["steps"]), "prompts": prompts, "clusters": [], "sources": [], "audio": "none",
                "note": "base small-sfx at the production setting (run_sidecar TIERS['default'] + fast_load) with prompt "
                        "candidate %s; judged clips are the 0.6 s _extract window at normalize_loudness "
                        "target_for('normal') cut from the same raw generation as the natural-length clips in "
                        "natural/%s; no real audio" % (slot, slot),
                "slot": slot, "seeds": list(range(args.seeds)), "intensity": INTENSITY, "context": "default",
                "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in kw.items()},
                "presentation": {"route": "0.6 s _extract window, rms target_for(normal)",
                                 "natural": "production trim=natural, level=peak"},
                "median_ms": float(np.median(all_ms)), "median_ms_by_key": per_key, "seconds": secs,
                "static": static, "window_fallbacks": fbs, "fast_stats": g.fast_stats()}
        dump_json(info, os.path.join(rd, "route.json"))
        timing["slots"][slot] = {"seconds": secs, "median_ms": info["median_ms"], "static": static,
                                 "window_fallbacks": fbs}
        print("slot %-14s %d keys x %d seeds  %.1f s  median %.1f ms  static %s  fallbacks %d" % (
            slot, len(keys), args.seeds, secs, info["median_ms"],
            " ".join("%s %d" % kv for kv in sorted(static.items()) if kv[1]), fbs), flush=True)
    g.prompts = {}
    timing["total_s"] = time.perf_counter() - t0
    dump_json(timing, os.path.join(args.out_dir, "gen_timing.json"))
    print("done %.1f s total" % timing["total_s"], flush=True)
    return 0


def cmd_spaces(args):
    C = load_json(args.candidates)
    R = load_json(args.ranking)
    plan = []
    for style in STYLES:
        for key in SPACE_KEYS:
            order = [r["slot"] for r in R["keys"][key]["ranking"] if r["slot"].startswith(style + "_")]
            ranked = [r["slot"] for r in R["keys"][key]["ranking"]
                      if r["slot"].startswith(style + "_") and r.get("rank") is not None]
            best = (ranked or order or [style + "_0"])[0]
            plan.append((style, "outdoor_open", key, C["candidates"][key][best], best))
            for sp in SPACES:
                plan.append((style, sp, key, C["spaces"][style][key][sp], None))
    t0 = time.perf_counter()
    g, kw = build_generator()
    g.ensure_ready()
    tap = Tap(g)
    g.generate(CacheKey("shoot", "rifle", INTENSITY), variant=99)
    tap.raw.clear()
    print("generator ready %.1f s" % (time.perf_counter() - t0), flush=True)
    prompts, ms_all = {}, []
    for style, sp, key, p, slot in plan:
        d = os.path.join(args.out_dir, "spaces", style, sp)
        os.makedirs(d, exist_ok=True)
        ms, _fb = render(g, tap, key, p, range(args.seeds), d)
        ms_all += ms
        prompts.setdefault(style, {}).setdefault(sp, {})[key] = {"prompt": p, "from_slot": slot,
                                                                 "files": [fname(key, v) for v in range(args.seeds)]}
        print("%-11s %-12s %-15s %s" % (style, sp, key, p), flush=True)
    g.prompts = {}
    info = {"seeds": list(range(args.seeds)), "intensity": INTENSITY, "context_key": "default",
            "presentation": "production trim=natural, level=peak",
            "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in kw.items()},
            "median_ms": float(np.median(ms_all)), "seconds": time.perf_counter() - t0, "prompts": prompts}
    dump_json(info, os.path.join(args.out_dir, "spaces", "prompts.json"))
    print("spaces done %.1f s, median %.1f ms" % (info["seconds"], info["median_ms"]), flush=True)
    return 0


def cmd_water_prep(args):
    C = load_json(args.candidates)
    base = os.path.join(args.out_dir, "water_tiers")
    if os.path.isdir(base):
        shutil.rmtree(base)
    n = 0
    for slot in C["slots"]:
        for key in WATER_KEYS:
            et, mat = key.split("|")
            d = os.path.join(base, "%s__%s-%s" % (slot, et, mat))
            os.makedirs(d, exist_ok=True)
            for p in sorted(glob.glob(os.path.join(args.out_dir, "routes", slot, "%s-%s_*.wav" % (et, mat)))):
                shutil.copy2(p, os.path.join(d, os.path.basename(p)))
                n += 1
    print("copied %d water clips into %s" % (n, base))
    return 0


def cmd_extra(args):
    import vocab_eval as ve
    from validate_panel import judge_audiobox
    C = load_json(args.candidates)
    man, sets, meta = ve.load_refs(SimpleNamespace(manifest=os.path.join(args.refs, "manifest.json")), {})
    need_types = sorted({t for t, _s in NEAREST.values() if t})
    pos = {t: ve.pos_sets(meta, t, "0.6") for t in need_types}
    shapes = {n: ve.shape_rows(sets[n]) for t in need_types for n in pos[t]}
    vocab_keys = {ve.key_name(et, mat) for et, mat, t in ve.KEYS if t}
    keys = [k for k in C["keys"] if k in NEAREST]
    out, flat, owner = {}, [], []
    for key in keys:
        t, surf = NEAREST[key]
        rows, src = ve.shape_refs(meta, shapes, man, t, "0.6", surf or "") if t else ([], None)
        out[key] = {"ref_type": t, "ref_surface": surf, "ref_source": src, "ref_clips": len(rows),
                    "vocab_judged": key in vocab_keys, "slots": {}}
        for slot in C["slots"]:
            ps = [os.path.join(args.out_dir, "routes", slot, fname(key, v)) for v in range(args.seeds)]
            ps = [p for p in ps if os.path.isfile(p)]
            if not ps:
                continue
            clips = [ve.wav(p) for p in ps]
            rec = {"n": len(clips)}
            if rows:
                rec["shape"] = ve.shape_compare(ve.shape_rows(clips), rows)
            out[key]["slots"][slot] = rec
            if key not in vocab_keys and key != "equip|rifle":
                flat += clips
                owner += [(key, slot)] * len(clips)
    real_flat, real_owner = [], []
    for t in need_types:
        for n in pos[t]:
            real_flat += sets[n]
            real_owner += [t] * len(sets[n])
    t0 = time.perf_counter()
    r = judge_audiobox(flat + real_flat, ve.SR, args.bs)
    print("audiobox %.1f s for %d clips" % (time.perf_counter() - t0, len(flat) + len(real_flat)), flush=True)
    rows_ab = r["rows"]
    pq, pq_real = {}, {}
    for row, o in zip(rows_ab[:len(flat)], owner):
        pq.setdefault(o, []).append(float(row["PQ"]))
    for row, t in zip(rows_ab[len(flat):], real_owner):
        pq_real.setdefault(t, []).append(float(row["PQ"]))
    for (key, slot), v in pq.items():
        out[key]["slots"][slot]["pq"] = float(np.mean(v))
    for key in out:
        t = out[key]["ref_type"]
        out[key]["pq_real"] = float(np.mean(pq_real[t])) if t in pq_real else None
    dump_json({"refs": args.refs, "window": "0.6", "keys": out}, os.path.join(args.out_dir, "extra.json"))
    for key, row in out.items():
        print("%-22s ref %-12s %s" % (key, row["ref_type"], "  ".join(
            "%s %.2f/%s" % (s, v["shape"]["in_range_mean"] if "shape" in v else float("nan"),
                            "%.2f" % v["pq"] if "pq" in v else "-") for s, v in row["slots"].items())), flush=True)
    return 0


def cmd_static(args):
    from sarge.static_guard import is_static
    C = load_json(args.candidates)
    out = {}
    for key in C["keys"]:
        for slot in C["slots"]:
            flags = []
            for v in range(args.seeds):
                p = os.path.join(args.out_dir, "natural", slot, fname(key, v))
                if os.path.isfile(p):
                    c = AudioClip.from_wav(p)
                    flags.append(int(is_static(c.samples, c.sample_rate)[0]))
            out.setdefault(key, {})[slot] = {"n": len(flags), "static_rate": float(np.mean(flags)) if flags else None,
                                             "flags": flags}
    dump_json(out, os.path.join(args.out_dir, "static.json"))
    for key, by in out.items():
        bad = {s: r["static_rate"] for s, r in by.items() if r["static_rate"]}
        print("%-22s %s" % (key, " ".join("%s %.2f" % kv for kv in bad.items()) or "clean"))
    return 0


def binom_p(g, l):
    n = g + l
    if n == 0:
        return 1.0
    k = min(g, l)
    s = sum(math.comb(n, i) for i in range(k + 1)) / 2.0 ** n
    return float(min(1.0, 2.0 * s))


def vocab_pair(V, slot, ref, key):
    et, mat = key.split("|")
    a, b = "gen:%s|%s|%s" % (slot, et, mat), "gen:%s|%s|%s" % (ref, et, mat)
    ca, cb = V["consensus"].get(a, {}), V["consensus"].get(b, {})
    out = {}
    for fam, j in (ca.get("judges") or {}).items():
        jb = (cb.get("judges") or {}).get(fam)
        if not jb or jb["variant"] != j["variant"]:
            continue
        ha = np.array(V["generated"][a][j["variant"]]["hits"], dtype=bool)
        hb = np.array(V["generated"][b][j["variant"]]["hits"], dtype=bool)
        g, l = int((ha & ~hb).sum()), int((~ha & hb).sum())
        out[fam] = {"gain": g, "loss": l, "p": binom_p(g, l)}
    return out


def water_scores(W, boot=2000, seed=0):
    sel = W["selection"]
    chosen = sel["chosen"].get("0.6", {})
    rng = np.random.default_rng(seed)
    out = {}
    for row in sel["ranking"].get("0.6", []):
        g = row["set"]
        name = g.split("|", 1)[1]
        slot, _, k = name.partition("__")
        key = k.replace("-", "|", 1)
        per, mats = {}, []
        for fam, v in chosen.items():
            rec = W["generated"][v][g]
            h = np.array([x[1] for x in rec["pairs"]], dtype=float)
            idx = rng.integers(0, len(h), size=(boot, len(h)))
            mats.append(np.minimum(h[idx].mean(1) / max(rec["real"], 1e-12), 1.0))
            per[fam] = {"variant": v, "rate": rec["rate"], "real": rec["real"], "ratio": rec["ratio"]}
        cmin = np.min(np.vstack(mats), axis=0) if mats else np.array([np.nan])
        out[(slot, key)] = {"cons": row["consensus"], "cons_ci": [float(np.percentile(cmin, 2.5)),
                                                                  float(np.percentile(cmin, 97.5))],
                            "mean_ratio": row["mean_ratio"], "static": row["static"], "judges": per,
                            "interval": "seed bootstrap only (real rate fixed)"}
    return out, chosen


def num(x, d=-1.0):
    return d if x is None or (isinstance(x, float) and x != x) else float(x)


def cmd_rank(args):
    C = load_json(args.candidates)
    V = load_json(args.vocab)
    E = load_json(args.extra) if os.path.isfile(args.extra) else {"keys": {}}
    S = load_json(args.static) if os.path.isfile(args.static) else {}
    W, wchosen = ({}, {})
    if os.path.isfile(args.water):
        W, wchosen = water_scores(load_json(args.water))
    summ = V["summary"]
    res = {"meta": {"vocab": args.vocab, "water": args.water if W else None, "extra": args.extra,
                    "rule": {"vocab": "cons desc, then shape in-range desc, then mean ratio",
                             "water": "water_judge consensus desc, then shape in-range vs splash refs desc",
                             "shape": "shape in-range vs the nearest real type desc, then PQ",
                             "unranked": "no judge and no nearest real type"},
                    "water_judges": wchosen}, "keys": {}}
    wins = {"cur": 0, "library": 0, "descriptive": 0}
    for key in C["keys"]:
        vs = summ.get(key)
        ex = E["keys"].get(key, {})
        judged = vs is not None and any(s.get("cons") is not None for s in vs["settings"].values())
        wjudged = key in WATER_KEYS and any((slot, key) in W for slot in C["slots"])
        rows = []
        for slot in C["slots"]:
            r = {"slot": slot, "style": slot.split("_")[0], "prompt": C["candidates"][key][slot],
                 "static_rate": S.get(key, {}).get(slot, {}).get("static_rate")}
            if vs is not None and slot in vs["settings"]:
                s = vs["settings"][slot]
                cj = V["consensus"].get("gen:%s|%s" % (slot, key), {}).get("judges") or s.get("judges")
                r.update({"cons": s.get("cons"), "cons_ci": s.get("cons_ci"), "mean_ratio": s.get("mean_ratio"),
                          "judges": cj, "pq": s.get("pq"), "pq_real": s.get("pq_real"),
                          "shape_in_range": s.get("shape_in_range"), "shape_ref": vs.get("type")})
                if key.startswith("footstep|"):
                    r.update({"material": s.get("material"), "material_ratio": s.get("material_ratio"),
                              "material_judged": s.get("material_judged")})
                if s.get("reason"):
                    r["reason"] = s["reason"]
                if s.get("cons") is None:
                    r["clap_identity_top"] = V["generated"].get("_identity", {}).get("gen:%s|%s" % (slot, key))
            es = ex.get("slots", {}).get(slot, {})
            if "shape" in es and r.get("shape_in_range") is None:
                r["shape_in_range"] = es["shape"]["in_range_mean"]
                r["shape_ref"] = "%s (%s)" % (ex.get("ref_type"), ex.get("ref_source"))
                r["shape_detail"] = {k: v for k, v in es["shape"].items() if k != "in_range_mean"}
            if "pq" in es and r.get("pq") is None:
                r["pq"] = es["pq"]
                r["pq_real"] = ex.get("pq_real")
            if (slot, key) in W:
                w = W[(slot, key)]
                r.update({"cons": w["cons"], "cons_ci": w["cons_ci"], "mean_ratio": w["mean_ratio"],
                          "water_static": w["static"], "judges": w["judges"], "cons_source": "water_judge",
                          "interval_note": w["interval"]})
            rows.append(r)
        if judged or wjudged:
            mode = "water" if wjudged else "vocab"
            rows.sort(key=lambda r: (num(r.get("cons")), num(r.get("shape_in_range")), num(r.get("mean_ratio"))),
                      reverse=True)
            ranked = [r for r in rows if r.get("cons") is not None]
        elif any(r.get("shape_in_range") is not None for r in rows):
            mode = "shape"
            rows.sort(key=lambda r: (num(r.get("shape_in_range")), num(r.get("pq"))), reverse=True)
            ranked = [r for r in rows if r.get("shape_in_range") is not None]
        else:
            mode = "unranked"
            ranked = []
        for i, r in enumerate(ranked):
            r["rank"] = i + 1
        top = ranked[0] if ranked else None
        cur = next(r for r in rows if r["slot"] == "cur")
        entry = {"mode": mode, "ranking": rows, "cur": {k: cur.get(k) for k in ("cons", "cons_ci", "shape_in_range",
                                                                                  "pq", "rank")}}
        if top:
            entry["winner"] = top["slot"]
            ties = [r["slot"] for r in ranked if (num(r.get("cons")), num(r.get("shape_in_range"))) ==
                    (num(top.get("cons")), num(top.get("shape_in_range")))]
            entry["tied_at_top"] = ties
            wins[top["style"]] += 1
            if mode == "vocab":
                entry["top_vs_cur"] = vocab_pair(V, top["slot"], "cur", key)
        res["keys"][key] = entry
    res["style_wins"] = wins
    best = {st: [] for st in STYLES}
    beats = {st: 0 for st in STYLES}
    for e in res["keys"].values():
        if e["mode"] == "unranked":
            continue
        cr = e["cur"].get("rank") or 99
        for st in STYLES:
            rk = [r["rank"] for r in e["ranking"] if r["style"] == st and r.get("rank")]
            if rk:
                best[st].append(min(rk))
                beats[st] += int(min(rk) < cr)
    res["style_mean_rank"] = {st: {"keys": len(v), "mean_best_rank": float(np.mean(v)) if v else None}
                              for st, v in best.items()}
    res["style_best_beats_cur"] = beats
    dump_json(res, args.out)
    for key, e in res["keys"].items():
        print("%-22s %-8s cur %s" % (key, e["mode"], fmt_row(next(r for r in e["ranking"] if r["slot"] == "cur"))))
        for r in [r for r in e["ranking"] if r.get("rank")][:3]:
            print("    #%d %-14s %s  %s" % (r["rank"], r["slot"], fmt_row(r), r["prompt"]))
    print("style wins %s  best-beats-cur %s  mean best rank %s" % (wins, beats, res["style_mean_rank"]))
    print("saved -> %s" % args.out)
    return 0


def fmt_row(r):
    c = r.get("cons")
    ci = r.get("cons_ci") or [None, None]
    s = "cons %s" % ("%.2f [%.2f,%.2f]" % (c, ci[0], ci[1]) if c is not None and ci[0] is not None else "-")
    s += "  shape %s" % ("%.2f" % r["shape_in_range"] if r.get("shape_in_range") is not None else "-")
    s += "  PQ %s" % ("%.2f" % r["pq"] if r.get("pq") is not None else "-")
    if r.get("material") is not None:
        s += "  mat %.2f" % r["material"]
    if r.get("static_rate"):
        s += "  static %.2f" % r["static_rate"]
    return s


def parse(argv):
    ap = argparse.ArgumentParser(description="Prompt candidate search on the production generator")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("gen", "spaces", "water-prep", "extra", "static", "rank"):
        p = sub.add_parser(name)
        p.add_argument("--candidates", default=CANDS)
        p.add_argument("--out-dir", default=OUT)
        if name == "gen":
            p.add_argument("--seeds", type=int, default=12)
            p.add_argument("--slots", nargs="+", default=None)
            p.add_argument("--keys", nargs="+", default=None)
        if name == "spaces":
            p.add_argument("--seeds", type=int, default=4)
            p.add_argument("--ranking", default=os.path.join(OUT, "ranking.json"))
        if name == "static":
            p.add_argument("--seeds", type=int, default=12)
        if name == "extra":
            p.add_argument("--seeds", type=int, default=12)
            p.add_argument("--refs", default=os.path.join("data", "refs_heldout"))
            p.add_argument("--bs", type=int, default=16)
        if name == "rank":
            p.add_argument("--vocab", default=os.path.join(OUT, "vocab.json"))
            p.add_argument("--water", default=os.path.join(OUT, "water.json"))
            p.add_argument("--extra", default=os.path.join(OUT, "extra.json"))
            p.add_argument("--static", default=os.path.join(OUT, "static.json"))
            p.add_argument("--out", default=os.path.join(OUT, "ranking.json"))
    return ap.parse_args(argv)


def main(argv=None):
    args = parse(argv)
    fn = {"gen": cmd_gen, "spaces": cmd_spaces, "water-prep": cmd_water_prep, "extra": cmd_extra,
          "static": cmd_static, "rank": cmd_rank}[args.cmd]
    return fn(args)


if __name__ == "__main__":
    raise SystemExit(main())

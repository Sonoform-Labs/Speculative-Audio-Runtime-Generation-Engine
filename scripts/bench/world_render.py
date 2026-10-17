import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

import woosh_server as ws
from sarge import post, world
from sarge.prompts import build
from sarge.types import CacheKey, seed_for

OUT = ROOT / "data" / "_gate" / "feel" / "world"
SR = ws.SR
CUT = int(ws.END_CUT_S * SR)
CONTEXT = "default"

BALL = {"shape": "sphere", "ext": [0.6, 0.6, 0.6], "m": 1.0, "dyn": 1, "pm": "sarge_bouncy", "bnc": 1.0,
        "rm": "Default-Material", "name": "AutoCaptureBall_0"}
BOTTLE = {"shape": "capsule", "ext": [0.15, 0.5, 0.15], "m": 0.4, "dyn": 1, "name": "glass bottle"}
BARREL = {"shape": "capsule", "ext": [0.6, 0.9, 0.6], "m": 8.0, "dyn": 1, "name": "metal barrel"}
CRATE = {"shape": "box", "ext": [0.8, 0.8, 0.8], "m": 4.0, "dyn": 1, "name": "wooden crate"}
PANE = {"shape": "box", "ext": [1.6, 2.0, 0.08], "dyn": 0, "name": "window"}


def tile(mat):
    return {"shape": "box", "ext": [10.0, 0.2, 10.0], "dyn": 0, "tag": mat, "pm": "sarge_bouncy", "bnc": 1.0,
            "name": "Surface_" + mat}


def contact(a, b, vn):
    return {"a": a, "b": b, "vn": vn, "vt": 0.1, "n": [0.0, 1.0, 0.0]}


DEMO = [
    ("ball-metal", "impact", "metal", "hard", contact(BALL, tile("metal"), 18.8)),
    ("ball-stone", "impact", "stone", "hard", contact(BALL, tile("stone"), 18.8)),
    ("ball-wood", "impact", "wood", "hard", contact(BALL, tile("wood"), 18.8)),
    ("ball-grass", "impact", "grass", "hard", contact(BALL, tile("grass"), 18.8)),
    ("ball-glass", "impact", "glass", "hard", contact(BALL, tile("glass"), 18.8)),
    ("bottle-glass", "glass bottle hit", "glass", "hard", contact(BOTTLE, tile("glass"), 7.0)),
    ("bottle-gravel", "glass bottle hit", "gravel", "normal", contact(BOTTLE, tile("gravel"), 3.0)),
    ("barrel-gravel", "metal barrel hit", "gravel", "hard", contact(BARREL, tile("gravel"), 7.7)),
    ("crate-gravel", "wooden crate hit", "gravel", "normal", contact(CRATE, tile("gravel"), 4.0)),
    ("crate-barrel", "wooden crate hit", "stone", "normal", contact(CRATE, BARREL, 4.0)),
    ("window-break", "break", "window", "normal", {"a": PANE, "kind": "break"}),
]
ARMS = {"head": ("peak", False), "world": ("peak", True), "world_loud": ("loudness", True)}
RENDER_ARMS = ("head", "world")

BASES = {
    "ball": ("impact", "metal", "hard", "bounce;a=rubber,large,inflated,ball;b=metal,huge,static,floor;c=soft;r=short,metal"),
    "bottle": ("glass bottle hit", "gravel", "normal",
               "hit;a=glass,medium,hollow,rod;b=gravel,huge,static,floor;c=dull;r=ringing,glass"),
    "crate_barrel": ("wooden crate hit", "stone", "normal",
                     "hit;a=wood,large,hollow,box;b=metal,large,hollow,cylinder;c=crisp;r=long,metal"),
    "barrel": ("metal barrel hit", "gravel", "hard",
               "hit;a=metal,large,hollow,cylinder;b=gravel,huge,static,floor;c=soft;r=ringing,metal"),
    "crate": ("wooden crate hit", "gravel", "normal", "hit;a=wood,large,hollow,box;b=gravel,huge,static,floor;c=dull;r=dead,wood"),
}
METRIC_WORD = {"cent": "centroid of the first 50 ms (Hz)", "t40": "peak to -40 dB (s)",
               "low": "share of energy 20-300 Hz"}
CELLS = [
    ("contact", "ball", "c", 0, "soft", "sharp", "cent"),
    ("contact", "bottle", "c", 0, "soft", "sharp", "cent"),
    ("contact", "crate_barrel", "c", 0, "soft", "sharp", "cent"),
    ("contact", "bottle", "c", 0, "dull", "crisp", "cent"),
    ("contact", "crate", "c", 0, "dull", "crisp", "cent"),
    ("ring", "ball", "r", 0, "dead", "short", "t40"),
    ("ring", "ball", "r", 0, "short", "ringing", "t40"),
    ("ring", "ball", "r", 0, "ringing", "long", "t40"),
    ("ring", "bottle", "r", 0, "dead", "short", "t40"),
    ("ring", "bottle", "r", 0, "short", "ringing", "t40"),
    ("ring", "bottle", "r", 0, "ringing", "long", "t40"),
    ("size", "ball", "a", 1, "small", "large", "low"),
    ("size", "barrel", "a", 1, "small", "large", "low"),
    ("size", "bottle", "a", 1, "small", "large", "low"),
    ("build", "barrel", "a", 2, "solid", "hollow", "t40"),
    ("build", "bottle", "a", 2, "solid", "hollow", "t40"),
    ("build", "crate", "a", 2, "solid", "hollow", "t40"),
    ("build", "ball", "a", 2, "inflated", "solid", "cent"),
]
PASS_OF = 5


def dump(obj, p):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1)


def load_json(p):
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def with_field(phys, field, idx, value):
    parts = phys.split(";")
    for i, p in enumerate(parts):
        name, _, val = p.partition("=")
        if name == field:
            vals = val.split(",")
            vals[idx] = value
            parts[i] = "%s=%s" % (name, ",".join(vals))
            return ";".join(parts)
    raise ValueError("no field %s in %s" % (field, phys))


def key_for(event, material, intensity, phys_json=None, world_on=True):
    b = world.bucket(phys_json, event) if world_on and phys_json else ""
    return CacheKey(event, material, intensity, CONTEXT, b)


def demo_key(name, arm):
    for n, ev, mat, inten, phys in DEMO:
        if n == name:
            return key_for(ev, mat, inten, phys, ARMS[arm][1])
    raise KeyError(name)


def open_gen(spec=None):
    opts = dict(ws.OPTIONS, **ws.ADOPTED["dflow"])
    frames = ws.parse_frames(spec)
    gen = ws.make_gen("dflow", frames["other"], 4, "dopri5", **opts)
    ws.prepare(gen, frames)
    return gen, frames


def prompt_of(key):
    return build(key.event_type, key.material, key.intensity, key.context, key.phys)


def render(gen, frames, key, variant=0, seed=None):
    prompt = prompt_of(key)
    s = seed_for(key, int(variant)) if seed is None else int(seed)
    t = time.perf_counter()
    raw = ws.render(gen, prompt, s, frames=ws.frames_for(key.event_type, frames))
    return raw, prompt, s, (time.perf_counter() - t) * 1000.0


def finish(raw, key, level="peak", amp="full", stereo=True):
    return ws.finish_key(raw, key, amp, stereo, level=level)


def mono(x):
    a = np.asarray(x, dtype=np.float64)
    return a if a.ndim == 1 else a.mean(axis=1)


def body(raw):
    return raw[:max(1, len(raw) - CUT)]


def onset(m):
    env = post.envelope(m, SR)
    if env.size == 0 or not np.any(env):
        return 0
    return int(np.flatnonzero(env >= env.max() * 0.1)[0])


def centroid50(m):
    i = onset(m)
    w = m[i:i + int(0.05 * SR)]
    if w.size < 16 or not np.any(w):
        return float("nan")
    p = np.abs(np.fft.rfft(w * np.hanning(w.size), n=8192)) ** 2
    f = np.fft.rfftfreq(8192, 1.0 / SR)
    band = f >= 20.0
    return float(np.sum(f[band] * p[band]) / max(np.sum(p[band]), 1e-30))


def low300(m):
    p = np.abs(np.fft.rfft(m)) ** 2
    f = np.fft.rfftfreq(m.size, 1.0 / SR)
    band = f >= 20.0
    return float(np.sum(p[band & (f < 300.0)]) / max(np.sum(p[band]), 1e-30))


def metrics(x, hp=False):
    m = mono(x)
    if hp:
        from sarge.loudness import dc_highpass
        m = np.asarray(dc_highpass(m.astype(np.float32), SR), dtype=np.float64)
    return {"t40": post.decay_time(m, SR), "cent": centroid50(m), "low": low300(m), "len": m.size / SR}


def cell_id(c):
    return "%s|%s|%s>%s" % (c[0], c[1], c[4], c[5])


WORD_TABLES = ("CONTACT_WORDS", "RING_WORDS", "SIZE_WORD", "BUILD_WORDS")


def use_words(over=None, order="late"):
    import sarge.prompts as P
    over = over or {}
    if not hasattr(P, "_world_render_physical"):
        P._world_render_physical = P.physical
        world._world_render_tables = {n: dict(getattr(world, n)) for n in WORD_TABLES}
    for n in WORD_TABLES:
        setattr(world, n, dict(world._world_render_tables[n], **over.get(n, {})))
    if order == "late":
        P.physical = P._world_render_physical
        return

    def early(ev, intensity, S, phys):
        w = world.prompt_words(phys, noun=P.event_noun(ev), intensity=intensity)
        if w["kind"] == "break":
            return P._world_render_physical(ev, intensity, S, phys)
        return P._join(w["head"], w["contact"], w["ring"], w["action"], "foley, close-distance", S["foley"])

    P.physical = early


def run_cell(gen, frames, c, variants):
    group, base, field, idx, lo, hi, metric = c
    ev, mat, inten, phys = BASES[base]
    base_key = CacheKey(ev, mat, inten, CONTEXT, phys)
    members = {}
    for side, val in (("lo", lo), ("hi", hi)):
        k = CacheKey(ev, mat, inten, CONTEXT, with_field(phys, field, idx, val))
        members[side] = {"phys": k.phys, "prompt": prompt_of(k), "raw": [], "chain": [], "gen_ms": []}
        for v in variants:
            raw, prompt, s, ms = render(gen, frames, k, seed=seed_for(base_key, v))
            members[side]["raw"].append(metrics(body(raw), hp=True))
            members[side]["chain"].append(metrics(finish(raw, k)))
            members[side]["gen_ms"].append(ms)
    out = {"group": group, "base": base, "field": field, "lo": lo, "hi": hi, "metric": metric, "members": members}
    for stage in ("raw", "chain"):
        lo_v = [r[metric] for r in members["lo"][stage]]
        hi_v = [r[metric] for r in members["hi"][stage]]
        moved = [bool(h > l) for l, h in zip(lo_v, hi_v)]
        out[stage] = {"lo_median": float(np.median(lo_v)), "hi_median": float(np.median(hi_v)),
                      "moved": int(sum(moved)), "pass": int(sum(moved)) >= PASS_OF}
    return out


def cell_line(out, n):
    return "%-8s %-13s %-17s %-5s raw %8.3f -> %8.3f moved %d/%d %s | chain %8.3f -> %8.3f moved %d/%d" % (
        out["group"], out["base"], "%s>%s" % (out["lo"], out["hi"]), out["metric"], out["raw"]["lo_median"],
        out["raw"]["hi_median"], out["raw"]["moved"], n, "PASS" if out["raw"]["pass"] else "FAIL",
        out["chain"]["lo_median"], out["chain"]["hi_median"], out["chain"]["moved"], n)


def cmd_words(a):
    gen, frames = open_gen(a.frames)
    variants = list(range(a.first, a.first + a.seeds))
    res = {"meta": {"variants": variants, "pass_of": PASS_OF, "metrics": METRIC_WORD,
                    "raw_stage": "raw render minus the final END_CUT_S, through the chain's 30 Hz dc_highpass only",
                    "chain": "finish(raw, event, intensity, 'full', stereo=True, level='peak', shape=world.shape(phys))",
                    "tables": {n: getattr(world, n) for n in WORD_TABLES}},
           "cells": {}}
    for c in CELLS:
        out = run_cell(gen, frames, c, variants)
        res["cells"][cell_id(c)] = out
        print(cell_line(out, len(variants)), flush=True)
    groups = {}
    for cid, c in res["cells"].items():
        groups.setdefault(c["group"], []).append(c["raw"]["pass"])
    res["groups"] = {g: {"cells": len(v), "passed": int(sum(v)), "pass": all(v)} for g, v in groups.items()}
    for g, v in res["groups"].items():
        print("group %-8s %d of %d cells pass -> %s" % (g, v["passed"], v["cells"], "PASS" if v["pass"] else "FAIL"))
    dump(res, OUT / ("words_%s.json" % a.tag))
    print("saved", OUT / ("words_%s.json" % a.tag))
    return 0


def cmd_explore(a):
    spec = load_json(a.spec)
    gen, frames = open_gen(a.frames)
    variants = list(range(a.first, a.first + a.seeds))
    res = {"meta": {"variants": variants, "spec": spec, "pass_of": PASS_OF}, "runs": {}}
    for order in spec.get("orders", ["late"]):
        for group, cands in spec["groups"].items():
            cells = [c for c in CELLS if c[0] == group]
            for cand in [{"name": "cur"}] + cands:
                use_words({n: cand[n] for n in WORD_TABLES if n in cand}, order)
                rows = [run_cell(gen, frames, c, variants) for c in cells]
                key = "%s|%s|%s" % (order, group, cand["name"])
                res["runs"][key] = {"passed": sum(r["raw"]["pass"] for r in rows), "cells": len(rows),
                                    "moved": sum(r["raw"]["moved"] for r in rows),
                                    "rows": [{k: r[k] for k in ("base", "lo", "hi", "metric", "raw", "chain")} |
                                             {"prompts": [r["members"]["lo"]["prompt"], r["members"]["hi"]["prompt"]]}
                                             for r in rows]}
                print("%-6s %-8s %-5s cells %d/%d moved %2d/%d  %s" % (
                    order, group, cand["name"], res["runs"][key]["passed"], len(rows), res["runs"][key]["moved"],
                    len(rows) * len(variants), " ".join("%s:%d" % (r["base"], r["raw"]["moved"]) for r in rows)),
                    flush=True)
    use_words()
    dump(res, OUT / ("explore_%s.json" % a.tag))
    print("saved", OUT / ("explore_%s.json" % a.tag))
    return 0


def raw_path(name, arm, v):
    return OUT / "raw" / name / ("%s_%02d.npy" % (arm, v))


def cmd_demo(a):
    gen, frames = open_gen(a.frames)
    meta = {"variants": a.variants, "identity_variants": a.identity, "pairs": {}}
    for name, ev, mat, inten, phys in DEMO:
        n = max(a.variants, a.identity if name == "window-break" else 0)
        meta["pairs"][name] = {"event": ev, "material": mat, "intensity": inten, "phys_json": phys, "arms": {}}
        for arm in RENDER_ARMS:
            k = demo_key(name, arm)
            rows = []
            for v in range(n):
                t = time.perf_counter()
                raw = ws.render_key(gen, k, v, frames)
                ms = (time.perf_counter() - t) * 1000.0
                p = raw_path(name, arm, v)
                p.parent.mkdir(parents=True, exist_ok=True)
                np.save(p, np.ascontiguousarray(raw, dtype=np.float32))
                rows.append({"variant": v, "seed": seed_for(k, v), "gen_ms": ms, "frames": int(raw.shape[0])})
            meta["pairs"][name]["arms"][arm] = {"phys": k.phys, "prompt": prompt_of(k), "renders": rows}
            print("%-14s %-6s %2d renders, median %.1f ms  %s" % (name, arm, n, float(np.median([r["gen_ms"] for r in rows])),
                                                               prompt_of(k)), flush=True)
    dump(meta, OUT / "demo_meta.json")
    print("saved", OUT / "demo_meta.json")
    return 0


def finished(name, arm, v):
    level, world_on = ARMS[arm]
    raw = np.load(raw_path(name, "world" if world_on else "head", v))
    return finish(raw, demo_key(name, arm), level=level)


def judge_cut(y):
    from math import gcd
    from scipy.signal import resample_poly
    from sarge.generators.stable_audio import _extract
    from sarge.loudness import normalize_loudness, target_for
    m = mono(y)
    g = gcd(SR, 44100)
    m = resample_poly(m, 44100 // g, SR // g).astype(np.float32)
    n = int(0.6 * 44100)
    if m.shape[0] < n:
        m = np.pad(m, (0, n - m.shape[0]))
    return normalize_loudness(_extract(m, 44100, 0.6), target_rms=target_for("normal")).astype(np.float32)


def cmd_routes(a):
    from sarge.types import AudioClip
    meta = load_json(OUT / "demo_meta.json")
    name = "window-break"
    ev, mat = meta["pairs"][name]["event"], meta["pairs"][name]["material"]
    for arm, route in (("head", "world_head"), ("world", "world_phys")):
        d = OUT / "routes" / route
        d.mkdir(parents=True, exist_ok=True)
        for v in range(a.identity):
            w = judge_cut(finished(name, arm, v))
            AudioClip(w, 44100).to_wav(str(d / ("%s-%s_%02d.wav" % (ev, mat, v))))
        info = meta["pairs"][name]["arms"][arm]
        dump({"clusters": [], "sources": [], "steps": 4, "prompts": {"%s|%s" % (ev, mat): info["prompt"]},
              "note": "Woosh DFlow through woosh_server.render_key and finish_key (--amp --stereo, peak level), %s; "
                      "judged clips are the mono 0.6 s _extract window at normalize_loudness target_for('normal') of the "
                      "finished clip, resampled to 44.1 kHz; variants 0-%d from seed_for" % (
                          "physics key " + info["phys"] if info["phys"] else "HEAD key", a.identity - 1)},
             d / "route.json")
        print("route %s: %d clips -> %s" % (route, a.identity, d))
    return 0


def cmd_check_served(a):
    from sarge.types import AudioClip
    served = AudioClip.from_wav(a.served, keep_channels=True).samples
    y = finished("ball-metal", "head", 0)
    q = np.round(np.clip(y, -1.0, 1.0) * 32767.0) / 32767.0
    same = served.shape == q.shape and float(np.max(np.abs(served - q))) <= 1.0 / 32767.0 + 1e-9
    print("served %s vs world_render head v0 %s, max abs %s, match %s" % (
        served.shape, y.shape, float(np.max(np.abs(served - q))) if served.shape == q.shape else "shape differs", same))
    return 0 if same else 1


def level_stats(y):
    from sarge import loudness as L
    st = np.asarray(y, dtype=np.float64)
    st = np.stack([st, st], axis=1) if st.ndim == 1 else st
    m = mono(st)
    pk = float(np.max(np.abs(st)))
    lufs = float(L.momentary_max(st.astype(np.float32), SR))
    rms_v = float(np.sqrt(np.mean(m ** 2)))
    return {"lufs": lufs, "peak_db": 20.0 * np.log10(max(pk, 1e-9)), "plr": 20.0 * np.log10(max(pk, 1e-9)) - lufs,
            "crest_db": 20.0 * np.log10(max(pk, 1e-9) / max(rms_v, 1e-12)), "t40": post.decay_time(m, SR),
            "len": m.size / SR}


def cap_acted(raw, key):
    from sarge.generators.stable_audio import _natural
    from sarge.loudness import dc_highpass
    sh = world.shape(key.phys)
    y = np.ascontiguousarray(body(raw[:, 0] if raw.ndim == 2 and raw.shape[1] == 1 else raw), dtype=np.float32)
    y = _natural(dc_highpass(y, SR).astype(np.float32), SR, floor_db=float(sh["floor_db"]))
    return post.decay_time(y, SR), post.decay_time(y, SR) > 1.25 * sh["t40_s"]


def cmd_analyse(a):
    meta = load_json(OUT / "demo_meta.json")
    res = {"decay": {}, "ball": {}, "raw_peak": {}}
    print("decay targets after the full chain (--world --amp --stereo, peak level), variants 0-%d" % (a.decay_variants - 1))
    rings = {}
    for name, ev, mat, inten, phys in DEMO:
        k = demo_key(name, "world")
        sh = world.shape(k.phys)
        ring = world.parse(k.phys)["r"][0]
        rows = []
        for v in range(a.decay_variants):
            raw = np.load(raw_path(name, "world", v))
            y = finished(name, "world", v)
            yl = finished(name, "world_loud", v)
            pre, acted = cap_acted(raw, k)
            t = post.decay_time(mono(y), SR)
            rows.append({"variant": v, "t40": t, "t40_loud": post.decay_time(mono(yl), SR), "ratio": t / sh["t40_s"],
                         "in_range": 0.67 <= t / sh["t40_s"] <= 1.5, "pre_cap_t40": pre, "cap_acted": bool(acted),
                         "len": y.shape[0] / SR})
            rings.setdefault(ring, []).append(rows[-1])
        res["decay"][name] = {"ring": ring, "target": sh["t40_s"], "rows": rows}
        print("%-14s %-8s target %.2f  t40 %s  in range %d/%d  cap acted %d  pre-cap t40 %s" % (
            name, ring, sh["t40_s"], " ".join("%.2f" % r["t40"] for r in rows), sum(r["in_range"] for r in rows),
            len(rows), sum(r["cap_acted"] for r in rows), " ".join("%.2f" % r["pre_cap_t40"] for r in rows)))
    res["by_ring"] = {}
    for ring, rows in sorted(rings.items(), key=lambda kv: world.RING.index(kv[0])):
        n_in = sum(r["in_range"] for r in rows)
        acted = [r for r in rows if r["cap_acted"]]
        res["by_ring"][ring] = {"clips": len(rows), "in_range": n_in, "share": n_in / len(rows),
                                "acted": len(acted), "acted_in_range": sum(r["in_range"] for r in acted),
                                "short_of_target": sum(r["ratio"] < 0.67 for r in rows),
                                "over_target": sum(r["ratio"] > 1.5 for r in rows)}
        b = res["by_ring"][ring]
        print("ring %-8s %2d clips, in 0.67-1.5x %2d (%.0f%%), below %d, above %d; cap acted on %d, %d of those in range%s" % (
            ring, b["clips"], b["in_range"], 100 * b["share"], b["short_of_target"], b["over_target"], b["acted"],
            b["acted_in_range"], "" if ring == "long" else ("  PASS" if b["share"] >= 0.8 else "  FAIL")))
    dd = [r for g in ("dead", "short", "ringing") for r in rings.get(g, [])]
    res["dead_short_ringing"] = {"clips": len(dd), "in_range": sum(r["in_range"] for r in dd)}
    print("dead+short+ringing: %d of %d in range (%.0f%%)" % (res["dead_short_ringing"]["in_range"], len(dd),
                                                             100.0 * res["dead_short_ringing"]["in_range"] / max(1, len(dd))))
    print()
    print("ball on metal, hard (bucket for every logged speed: %s)" % sorted(set(
        world.bucket(contact(BALL, tile("metal"), v), "impact") for v in a.speeds)))
    for arm in ARMS:
        rows = [level_stats(finished("ball-metal", arm, v)) for v in range(a.ball_variants)]
        res["ball"][arm] = {"rows": rows, "median": {k: float(np.median([r[k] for r in rows])) for k in rows[0]}}
        md = res["ball"][arm]["median"]
        print("%-10s M400 %6.1f LUFS (%s)  t40 %.2f s (%s)  len %.2f s  peak %5.1f dBFS  PLR %4.1f dB  crest %4.1f dB" % (
            arm, md["lufs"], " ".join("%.1f" % r["lufs"] for r in rows), md["t40"], " ".join("%.2f" % r["t40"] for r in rows),
            md["len"], md["peak_db"], md["plr"], md["crest_db"]))
    print()
    for name, ev, mat, inten, phys in DEMO:
        for arm in RENDER_ARMS:
            pk = [float(np.max(np.abs(body(np.load(raw_path(name, arm, v)))))) for v in range(a.ball_variants)]
            ms = [r["gen_ms"] for r in meta["pairs"][name]["arms"][arm]["renders"][:a.ball_variants]]
            res["raw_peak"]["%s|%s" % (name, arm)] = {"peaks": pk, "gen_ms": ms}
        print("%-14s raw peak head %s | world %s" % (name, " ".join("%.3f" % p for p in res["raw_peak"][name + "|head"]["peaks"]),
                                                    " ".join("%.3f" % p for p in res["raw_peak"][name + "|world"]["peaks"])))
    dump(res, OUT / "analyse.json")
    print("saved", OUT / "analyse.json")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="render (event, material, intensity, phys) requests through the Woosh "
                                             "worker's own build(), render() and finish()")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("words")
    p.add_argument("--tag", default="v1")
    p.add_argument("--seeds", type=int, default=6)
    p.add_argument("--first", type=int, default=0)
    p.add_argument("--frames", default=None)
    p = sub.add_parser("explore")
    p.add_argument("--spec", required=True)
    p.add_argument("--tag", default="x1")
    p.add_argument("--seeds", type=int, default=6)
    p.add_argument("--first", type=int, default=6)
    p.add_argument("--frames", default=None)
    p = sub.add_parser("demo")
    p.add_argument("--variants", type=int, default=6)
    p.add_argument("--identity", type=int, default=20)
    p.add_argument("--frames", default=None)
    p = sub.add_parser("analyse")
    p.add_argument("--decay-variants", type=int, default=4)
    p.add_argument("--ball-variants", type=int, default=6)
    p.add_argument("--speeds", type=float, nargs="+", default=[12.2, 15.0, 18.8, 22.0, 25.3])
    p = sub.add_parser("routes")
    p.add_argument("--identity", type=int, default=20)
    p = sub.add_parser("check-served")
    p.add_argument("--served", default=str(ROOT / "data" / "_gate" / "feel" / "served_ball_metal_hard.wav"))
    a = ap.parse_args(argv)
    return {"words": cmd_words, "explore": cmd_explore, "demo": cmd_demo, "analyse": cmd_analyse, "routes": cmd_routes,
            "check-served": cmd_check_served}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())

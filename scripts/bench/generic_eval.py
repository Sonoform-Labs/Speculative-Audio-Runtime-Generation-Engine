import os
import sys

BENCH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BENCH)

import _bootstrap

import argparse
import base64
import hashlib
import html
import json
import math
import re
import shutil
import time
from math import gcd

import numpy as np

from sarge.generators.routed import side_for
from sarge.prompts import TEMPLATE as STYLE_TEMPLATE, TEMPLATES, TT_OFF, TT_ON
from sarge.types import AudioClip, CacheKey

ROOT = os.path.dirname(os.path.dirname(BENCH))
SCRIPTS = os.path.join(ROOT, "scripts")
GATE = os.path.join(ROOT, "data", "_gate", "generic")
WORK = os.path.join(GATE, "_work")
PAGE = os.path.join(GATE, "page")
WATER_DIR = os.path.join(GATE, "_water")
SEEDS = 40
TAKES = 4
JSR = 44100
JLEN = 0.6
INTENSITY, CONTEXT = "normal", "default"
DEMO = [("shoot", "pistol"), ("shoot", "rifle"), ("break", "window"), ("bullet impact", "glass"),
        ("footstep", "gravel"), ("footstep", "stone"), ("footstep", "wood"), ("footstep", "metal"),
        ("footstep", "water"), ("splash", "water"), ("punch", "body"), ("reload", "rifle")]
UNSEEN = [("dog_bark", "dog"), ("door_creak", "wood"), ("lightsaber_swing", "plasma"), ("grappling_hook_fire", "rope"),
          ("campfire_crackle", "wood"), ("keyboard_click", "plastic"), ("ice_crack", "ice"),
          ("footsteps_Concrete", "concrete"), ("PlayerFootstep", "snow"), ("shotgun_fire", "shotgun"),
          ("bow_release", "bow"), ("laser_zap", "laser"), ("sword_clash", "steel"), ("car_door_slam", "car"),
          ("coin_pickup", "gold"), ("explosion_large", "debris"), ("glass_shatter", "glass"), ("water_drip", "water"),
          ("rain_on_roof", "metal"), ("creature_growl", "monster"), ("car_horn", "car"), ("Pistol_Shot", "9mm"),
          ("melee_hit", "body")]
KEYS = [(et, mat, "demo") for et, mat in DEMO] + [(et, mat, "unseen") for et, mat in UNSEEN]
VOCAB = {"shoot|pistol": "gunshot_pistol", "shoot|rifle": "gunshot_rifle", "break|window": "glass_break",
         "bullet impact|glass": "glass_break", "footstep|gravel": "footstep", "footstep|stone": "footstep",
         "footstep|wood": "footstep", "footstep|metal": "footstep", "footstep|water": "footstep",
         "splash|water": "splash", "punch|body": "punch", "dog_bark|dog": "dog_bark", "door_creak|wood": "door_open",
         "keyboard_click|plastic": "ui_click", "footsteps_Concrete|concrete": "footstep",
         "PlayerFootstep|snow": "footstep", "shotgun_fire|shotgun": "gunshot_shotgun", "sword_clash|steel": "sword",
         "car_door_slam|car": "door_slam", "explosion_large|debris": "explosion", "glass_shatter|glass": "glass_break",
         "car_horn|car": "horn", "Pistol_Shot|9mm": "gunshot_pistol", "melee_hit|body": "punch"}
APPROX = {"bullet impact|glass": "glass-break references for a bullet through a pane",
          "keyboard_click|plastic": "mouse and button clicks stand in for a keyboard",
          "car_door_slam|car": "house door slams stand in for a car door",
          "explosion_large|debris": "fireworks stand in for an explosion",
          "melee_hit|body": "foley punches stand in for a melee hit",
          "rain_on_roof|metal": "water drops stand in for rain on a roof"}
NO_JUDGE = {"reload|rifle": "gun mechanism references too thin on the held-out split (6 clips, 4 sources)",
            "lightsaber_swing|plasma": "no reference class", "grappling_hook_fire|rope": "no reference class",
            "campfire_crackle|wood": "crackling fire is a negatives-only set", "ice_crack|ice": "no reference class",
            "bow_release|bow": "no reference class", "laser_zap|laser": "no reference class",
            "coin_pickup|gold": "shell-drop references too thin (3 clips)",
            "creature_growl|monster": "no reference class"}
WATER = {"footstep|water": "footstep-water", "splash|water": "splash-water", "water_drip|water": "waterdrip-water",
         "rain_on_roof|metal": "rainonroof-metal"}
WEAPON_CLASS = {"shoot|pistol", "shoot|rifle", "reload|rifle", "shotgun_fire|shotgun", "Pistol_Shot|9mm"}
ARMS = {"phrasebook": ("tuned", "families", "full"), "phrasebook_words": ("tuned", "words", "full"),
        "generic": ("generic", "families", "full"), "generic_words": ("generic", "words", "full"),
        "generic_class": ("generic", "class", "full"), "generic_all": ("generic", "words", "auto"),
        "generic_bare": ("generic_bare", "families", "full")}
PARTNER = {"phrasebook_words": "phrasebook", "generic_words": "generic", "generic_class": "generic_words",
           "generic_bare": "generic"}
MAIN_ARMS = ("phrasebook", "generic", "generic_all")
ARM_TEXT = {"phrasebook": "shipped flags: --amp --detailed-prompts (tuned phrasebook prompts, substring families route, "
                          "family amplifier)",
            "phrasebook_words": "phrasebook prompts with --route words (guns picked by whole words), family amplifier",
            "generic": "--prompts generic: prompt from the request words only, families route, family amplifier",
            "generic_words": "--prompts generic --route words, family amplifier",
            "generic_class": "--prompts generic --route class (the game tags firearms class weapon), family amplifier",
            "generic_all": "--prompts generic --route words --amp auto (the whole generic path)",
            "generic_bare": "diagnostic, no flag: the --prompts generic text without the 'TrackType: SFX, ' prefix on SA3 "
                            "(Woosh never gets the prefix, so its audio is the generic arm's), families route, family "
                            "amplifier"}
OLD_ARMS = tuple(ARMS)
for _n, (_text, _tt, _space) in TEMPLATES.items():
    for _sfx, _word in ((TT_OFF, "without"), (TT_ON, "with")):
        ARMS[_n + _sfx] = (STYLE_TEMPLATE + _n + _sfx, "families", "full")
        ARM_TEXT[_n + _sfx] = ("--prompts template:%s%s: the universal template %r, %s the 'TrackType: SFX, ' prefix on "
                               "SA3 (Woosh never gets it), families route, family amplifier" % (_n, _sfx, _text, _word))
    PARTNER[_n + TT_ON] = _n + TT_OFF
TMPL_ARMS = ("phrasebook", "generic") + tuple(a for a in ARMS if a not in OLD_ARMS)
RUNS = {"old": (OLD_ARMS, MAIN_ARMS), "templates": (TMPL_ARMS, ("phrasebook", "generic") + tuple(
    n + TT_OFF for n in TEMPLATES))}
RUN, WATER_ARMS = RUNS["old"]
WATER_ALIAS = {n + TT_ON: n + TT_OFF for n in TEMPLATES}
TIER_LETTERS = "abcdefghijklmnopqrstuvwxyz"
MODEL_NAME = {"sa3": "SA3 medium", "woosh": "Woosh DFlow"}
STYLE_TO_SA3 = {"tuned": "detailed", "generic": "generic", "generic_bare": "generic"}
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
BOOT, BOOT_SEED = 4000, 20261017


def kid(et, mat):
    return "%s|%s" % (et, mat)


def slug(s):
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")


def key_of(et, mat, cls=""):
    return CacheKey(et, mat, INTENSITY, CONTEXT, "", cls)


def source(arm, et, mat):
    style, route, amp = ARMS[arm]
    cls = "weapon" if route == "class" and kid(et, mat) in WEAPON_CLASS else ""
    model = "sa3" if side_for(key_of(et, mat, cls), route) == "guns" else "woosh"
    if model == "woosh":
        style = woosh_style(style)
    return (model, style, amp)


def woosh_style(style):
    if style == "generic_bare":
        return "generic"
    if style.startswith(STYLE_TEMPLATE) and style[-len(TT_ON):] in (TT_ON, TT_OFF):
        return style[:-len(TT_ON)]
    return style


def src_name(src):
    return "%s-%s-%s" % (src[0], src[1].replace(":", "_"), src[2])


def water_tiers():
    return {arm: "%s_%s" % (TIER_LETTERS[i], arm) for i, arm in enumerate(WATER_ARMS)}


def water_arm(arm):
    return arm if arm in WATER_ARMS else WATER_ALIAS.get(arm)


def plan():
    out = {"sa3": {}, "woosh": {}}
    for et, mat, _g in KEYS:
        for arm in RUN:
            model, style, amp = source(arm, et, mat)
            out[model].setdefault((et, mat, style), set()).add(amp)
    return out


def clip_path(work, src, et, mat, v):
    return os.path.join(work, "clips", src_name(src), "%s_%02d.npy" % (slug(kid(et, mat)), v))


def cut_path(work, src, et, mat, v):
    return os.path.join(work, "cuts", src_name(src), "%s-%s_%02d.wav" % (et, mat, v))


def dump(obj, p):
    os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


def load_json(p):
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def mono(y):
    a = np.asarray(y, dtype=np.float64)
    return a if a.ndim == 1 else a.mean(axis=1)


def judge_cut(y, sr):
    from scipy.signal import resample_poly
    from sarge.generators.stable_audio import _extract
    from sarge.loudness import normalize_loudness, target_for
    m = mono(y)
    g = gcd(int(sr), JSR)
    m = resample_poly(m, JSR // g, int(sr) // g).astype(np.float32)
    n = int(JLEN * JSR)
    if m.shape[0] < n:
        m = np.pad(m, (0, n - m.shape[0]))
    return normalize_loudness(_extract(m, JSR, JLEN), target_rms=target_for(INTENSITY)).astype(np.float32)


def facts(y, sr):
    from sarge.post import decay_time
    from sarge.static_guard import is_static
    m = mono(y)
    return {"len": round(m.shape[0] / sr, 4), "t40": round(float(decay_time(m.astype(np.float32), sr)), 4),
            "peak": round(float(np.max(np.abs(y))), 4), "static": bool(is_static(m.astype(np.float32), sr)[0])}


def save_take(work, src, et, mat, v, y, sr, takes):
    y = np.ascontiguousarray(y, dtype=np.float32)
    if v < takes:
        p = clip_path(work, src, et, mat, v)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        np.save(p, y)
    AudioClip(judge_cut(y, sr), JSR).to_wav(cut_path(work, src, et, mat, v))
    return dict(facts(y, sr), sha1=hashlib.sha1(y.tobytes()).hexdigest())


def cmd_render_woosh(a):
    sys.path.insert(1, SCRIPTS)
    import woosh_server as ws
    opts = dict(ws.OPTIONS, **ws.ADOPTED["dflow"])
    frames = ws.parse_frames(None)
    t0 = time.perf_counter()
    gen = ws.make_gen("dflow", frames["other"], 4, "dopri5", **opts)
    ws.prepare(gen, frames)
    print("woosh dflow ready in %.1f s, %s" % (time.perf_counter() - t0, opts), flush=True)
    rows, jobs = [], sorted(plan()["woosh"].items(), key=lambda kv: (kv[0][2], kv[0][0], kv[0][1]))
    for (et, mat, style), amps in jobs:
        k = key_of(et, mat)
        prompt = ws.prompt_for(k, style)
        ms = []
        for v in range(a.seeds):
            t = time.perf_counter()
            raw = ws.render_key(gen, k, v, frames, style)
            ms.append((time.perf_counter() - t) * 1000.0)
            for amp in sorted(amps):
                y = ws.finish_key(raw, k, amp, True)
                rows.append(dict(save_take(a.work, ("woosh", style, amp), et, mat, v, y, ws.SR, a.takes), key=kid(et, mat),
                                 model="woosh", style=style, amp=amp, v=v, sr=ws.SR, prompt=prompt,
                                 gen_ms=round(ms[-1], 2)))
        print("woosh %-8s %-30s %s  median %.1f ms  %s" % (style, kid(et, mat), "+".join(sorted(amps)),
                                                          float(np.median(ms)), prompt), flush=True)
    dump({"meta": {"model": "woosh-dflow", "options": opts, "frames": frames, "seeds": a.seeds, "takes": a.takes,
                   "graph_error": gen.stats.get("graph_error"), "seconds": time.perf_counter() - t0},
          "rows": rows}, os.path.join(a.work, "render_woosh.json"))
    print("saved %d rows" % len(rows), flush=True)
    return 0


def cmd_render_sa3(a):
    sys.path.insert(1, SCRIPTS)
    import run_sidecar as rs
    from sarge.generators.routed import WARM_KEY
    from sarge.generators.stable_audio3 import StableAudio3Generator
    kw = dict(rs.TIERS["default"], fast_load=True, stereo=True, prompt_style="detailed", amp=True)
    kw.update(rs.HYBRID_GUNS)
    wc = rs.cache_dir_of(argparse.Namespace(weight_cache=None))
    if wc:
        kw["weight_cache"] = wc
    t0 = time.perf_counter()
    g = StableAudio3Generator(**kw)
    g.ensure_ready()
    g.generate(WARM_KEY, variant=0)
    g.generate(WARM_KEY, variant=0)
    print("sa3 medium ready in %.1f s" % (time.perf_counter() - t0), flush=True)
    rows, jobs = [], sorted(plan()["sa3"].items(), key=lambda kv: (kv[0][2], kv[0][0], kv[0][1]))
    jobs = [j for j in jobs if not a.styles or j[0][2] in a.styles]
    for (et, mat, style), amps in jobs:
        k = key_of(et, mat)
        g.prompt_style = STYLE_TO_SA3.get(style, style)
        g.prompts.clear()
        if style == "generic_bare":
            from sarge.prompts import generic
            g.prompts[str(k)] = generic(et, mat, INTENSITY, CONTEXT, "", "")
        prompt = g._prompt(k)
        for amp in sorted(amps):
            g.amp, g.amp_mode = True, amp
            ms, retries = [], 0
            for v in range(a.seeds):
                r0 = int(g.static_stats.get("static_retries", 0))
                t = time.perf_counter()
                clip = g.generate(k, variant=v)
                ms.append((time.perf_counter() - t) * 1000.0)
                rr = int(g.static_stats.get("static_retries", 0)) - r0
                retries += rr
                rows.append(dict(save_take(a.work, ("sa3", style, amp), et, mat, v, clip.samples, clip.sample_rate,
                                           a.takes), key=kid(et, mat), model="sa3", style=style, amp=amp, v=v,
                                 sr=int(clip.sample_rate), prompt=prompt, gen_ms=round(ms[-1], 2), static_retries=rr))
            print("sa3   %-8s %-30s %-4s median %.1f ms  static retries %d  %s" % (
                style, kid(et, mat), amp, float(np.median(ms)), retries, prompt), flush=True)
    dump({"meta": {"model": "sa3-medium", "kwargs": {k: v for k, v in kw.items()}, "seeds": a.seeds, "takes": a.takes,
                   "fast_stats": g.fast_stats(), "seconds": time.perf_counter() - t0},
          "rows": rows}, os.path.join(a.work, "render_sa3%s.json" % ("_" + "_".join(a.styles) if a.styles else "")))
    print("saved %d rows" % len(rows), flush=True)
    return 0


def render_rows(work):
    import glob
    rows = []
    for p in sorted(glob.glob(os.path.join(work, "render_*.json"))):
        rows += load_json(p)["rows"]
    return rows


def prompts_by_src(work):
    out = {}
    for r in render_rows(work):
        out[(r["key"], r["model"], r["style"])] = r["prompt"]
    return out


def cmd_verify(a):
    keyset = {kid(et, mat): (et, mat) for et, mat, _g in KEYS}
    res = {}
    for spec in a.served:
        arm, d = spec.split("=", 1)
        idx = load_json(os.path.join(d, "index.json"))
        rows = []
        for r in idx["rows"]:
            k = r.get("key")
            if "file" not in r or len(k) != 4 or k[2:] != [INTENSITY, CONTEXT]:
                continue
            em = kid(k[0], k[1])
            if em not in keyset:
                continue
            src = source(arm, *keyset[em])
            p = clip_path(a.work, src, k[0], k[1], 0)
            served = np.load(os.path.join(d, r["file"]))
            mine = np.load(p) if os.path.isfile(p) else None
            same_shape = mine is not None and mine.shape == served.shape
            diff = float(np.max(np.abs(mine - served))) if same_shape else None
            rows.append({"key": em, "model": src[0], "served_sr": r["sr"], "shape_served": list(served.shape),
                         "shape_mine": list(mine.shape) if mine is not None else None, "max_abs": diff,
                         "identical": bool(same_shape and diff == 0.0)})
            print("%-14s %-30s %-6s served %-14s mine %-14s max abs %s" % (
                arm, em, src[0], served.shape, mine.shape if mine is not None else None, diff), flush=True)
        res[arm] = {"compared": len(rows), "identical": sum(r["identical"] for r in rows), "rows": rows}
        print("%s: %d of %d served clips identical to the in-process render" % (
            arm, res[arm]["identical"], res[arm]["compared"]), flush=True)
    dump(res, os.path.join(a.work, "verify.json"))
    return 0 if all(v["identical"] == v["compared"] for v in res.values()) else 1


def link(src, dst):
    if os.path.exists(dst):
        os.remove(dst)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def route_keys(arm):
    out = []
    for et, mat, _g in KEYS:
        if not VOCAB.get(kid(et, mat)):
            continue
        if arm in PARTNER and source(arm, et, mat) == source(PARTNER[arm], et, mat):
            continue
        out.append((et, mat))
    return out


def route_for(arm, et, mat):
    p = arm
    while (et, mat) not in route_keys(p):
        if p not in PARTNER:
            return None
        p = PARTNER[p]
    return p


def cmd_routes(a):
    prompts = prompts_by_src(a.work)
    made = {}
    for arm in RUN:
        keys = route_keys(arm)
        d = os.path.join(a.work, "routes", arm)
        if os.path.isdir(d):
            shutil.rmtree(d)
        if not keys:
            continue
        os.makedirs(d)
        pr = {}
        for et, mat in keys:
            src = source(arm, et, mat)
            pr[kid(et, mat)] = prompts.get((kid(et, mat), src[0], src[1]))
            for v in range(a.seeds):
                link(cut_path(a.work, src, et, mat, v), os.path.join(d, "%s-%s_%02d.wav" % (et, mat, v)))
        dump({"clusters": [], "sources": [], "steps": None, "prompts": pr,
              "note": "%s; served path rendered in process (woosh_server.render_key + finish_key, StableAudio3Generator "
                      "with run_sidecar HYBRID_GUNS); judged clip = mono 0.6 s _extract window of the finished clip at "
                      "normalize_loudness target_for('normal'), 44.1 kHz; seeds 0-%d from seed_for" % (
                          ARM_TEXT[arm], a.seeds - 1)}, os.path.join(d, "route.json"))
        made[arm] = len(keys)
    wd = a.water_dir
    if os.path.isdir(wd):
        shutil.rmtree(wd)
    for arm, tier in water_tiers().items():
        d = os.path.join(wd, tier)
        os.makedirs(d)
        for et, mat, _g in KEYS:
            stem = WATER.get(kid(et, mat))
            if not stem:
                continue
            src = source(arm, et, mat)
            for v in range(a.seeds):
                link(cut_path(a.work, src, et, mat, v), os.path.join(d, "%s_%02d.wav" % (stem, v)))
    os.makedirs(os.path.join(a.work, "water_empty"), exist_ok=True)
    print("routes %s, water tiers %s" % (made, sorted(os.listdir(wd))), flush=True)
    return 0


def cmd_judge_vocab(a):
    import vocab_eval as ve
    ve.KEYS = [(et, mat, VOCAB[kid(et, mat)]) for et, mat, _g in KEYS if VOCAB.get(kid(et, mat))]
    tag = "_" + a.tag if a.tag else ""
    argv = ["--refs", a.refs, "--settings", "none", "--seeds", str(a.seeds), "--out",
            os.path.join(a.work, "vocab%s.json" % tag), "--listen", os.path.join(a.work, "vocab_listen%s.wav" % tag)]
    for arm in (a.arms or RUN):
        d = os.path.join(a.work, "routes", arm)
        if os.path.isdir(d):
            argv += ["--route", "%s=%s" % (arm, d)]
    return ve.main(argv)


def cmd_judge_water(a):
    import water_judge as wj
    return wj.main(["--ctl-lengths", "0.6", "--probe-json", os.path.join(a.work, "no_probe.json"), "--probe-dir",
                    os.path.join(a.work, "water_empty"), "--tier-dir", a.water_dir, "--out",
                    os.path.join(a.work, "water.json")])


def sign_two(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2.0 ** n)


def boot_delta(hits_a, hits_b, reals, rng):
    n = len(next(iter(hits_a.values())))
    idx = rng.integers(0, n, size=(BOOT, n))
    ca = np.full(BOOT, np.inf)
    cb = np.full(BOOT, np.inf)
    for f in hits_a:
        r = reals[f]
        ha, hb = np.asarray(hits_a[f], float), np.asarray(hits_b[f], float)
        ca = np.minimum(ca, np.minimum(ha[idx].mean(1) / r, 1.0))
        cb = np.minimum(cb, np.minimum(hb[idx].mean(1) / r, 1.0))
    d = ca - cb
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = 2.0 * min(float(np.mean(d <= 0)), float(np.mean(d >= 0)))
    return [float(lo), float(hi)], min(1.0, p)


def cons_of(hits, reals):
    return float(min(min(float(np.mean(hits[f])) / reals[f], 1.0) for f in hits))


def vocab_table(V, rng):
    cons, gen = V["consensus"], V["generated"]
    shape = V.get("shape", {})
    mat = V.get("material", {}).get("generated", {})
    out = {}
    for et, mat_, grp in KEYS:
        k = kid(et, mat_)
        if not VOCAB.get(k):
            continue
        arms = {}
        for arm in RUN:
            use = route_for(arm, et, mat_)
            g = "gen:%s|%s|%s" % (use, et, mat_) if use else None
            c = cons.get(g) if g else None
            if not c:
                continue
            row = {"route": use, "cons": c.get("cons"), "cons_ci": c.get("cons_ci"), "reason": c.get("reason"),
                   "judges": {f: {"variant": j["variant"], "rate": j["rate"], "real": j["real"], "ratio": j["ratio"]}
                              for f, j in (c.get("judges") or {}).items()},
                   "hits": {f: gen[g][j["variant"]]["hits"] for f, j in (c.get("judges") or {}).items()},
                   "shape_in_range": shape.get(g, {}).get("in_range_mean"),
                   "material": mat.get(g, {}).get("accuracy") if mat.get(g, {}).get("judged") else None}
            arms[arm] = row
        base = arms.get("phrasebook")
        for arm, row in arms.items():
            if arm == "phrasebook" or not base or row["cons"] is None or base["cons"] is None:
                continue
            reals = {f: j["real"] for f, j in base["judges"].items()}
            pairs = {}
            for f in base["hits"]:
                ha, hb = np.asarray(row["hits"][f], bool), np.asarray(base["hits"][f], bool)
                gn, ls = int((ha & ~hb).sum()), int((~ha & hb).sum())
                pairs[f] = {"gain": gn, "loss": ls, "p": sign_two(gn, ls)}
            ci, p = boot_delta(row["hits"], base["hits"], reals, rng)
            row["vs_phrasebook"] = {"delta": row["cons"] - base["cons"], "delta_ci": ci, "p_boot": p, "mcnemar": pairs,
                                    "floor_ok": bool(row["cons"] >= base["cons_ci"][0])}
        out[k] = {"group": grp, "type": VOCAB[k], "approx": APPROX.get(k), "arms": arms}
    return out


def water_table(W, rng):
    sel = W["selection"]
    ch = sel["chosen"].get("0.6") or {}
    val = W["validity"]
    gen = W["generated"]
    out = {}
    tiers = {arm: "gen:tier|" + water_tiers()[water_arm(arm)] for arm in RUN if water_arm(arm)}
    for k, stem in WATER.items():
        wk_ = "%s|%s" % tuple(stem.split("-"))
        arms = {}
        for arm, tg in tiers.items():
            hits, rates, reals = {}, {}, {}
            for f, v in ch.items():
                rec = gen.get(v, {}).get(tg)
                if not rec:
                    continue
                hs = sorted((int(p[0].split("#")[1]), int(p[1])) for p in rec["pairs"] if p[0].split("#")[0] == wk_)
                hits[f] = [h for _s, h in hs]
                reals[f] = val[v]["0.6"]["real"]
                rates[f] = float(np.mean(hits[f])) if hits[f] else float("nan")
            if not hits:
                continue
            c = cons_of(hits, reals)
            idx = rng.integers(0, len(next(iter(hits.values()))), size=(BOOT, len(next(iter(hits.values())))))
            cb = np.full(BOOT, np.inf)
            for f in hits:
                cb = np.minimum(cb, np.minimum(np.asarray(hits[f], float)[idx].mean(1) / reals[f], 1.0))
            arms[arm] = {"cons": c, "cons_ci": [float(x) for x in np.percentile(cb, [2.5, 97.5])], "rates": rates,
                         "reals": reals, "hits": hits, "variants": ch}
        base = arms.get("phrasebook")
        for arm, row in arms.items():
            if arm == "phrasebook" or not base:
                continue
            pairs = {}
            for f in base["hits"]:
                ha, hb = np.asarray(row["hits"][f], bool), np.asarray(base["hits"][f], bool)
                gn, ls = int((ha & ~hb).sum()), int((~ha & hb).sum())
                pairs[f] = {"gain": gn, "loss": ls, "p": sign_two(gn, ls)}
            ci, p = boot_delta(row["hits"], base["hits"], base["reals"], rng)
            row["vs_phrasebook"] = {"delta": row["cons"] - base["cons"], "delta_ci": ci, "p_boot": p, "mcnemar": pairs,
                                    "floor_ok": bool(row["cons"] >= base["cons_ci"][0])}
        out[k] = {"approx": APPROX.get(k), "arms": arms}
    return out


def pooled(vt, arm, group, rng):
    ks = [k for k, r in vt.items() if (group == "all" or r["group"] == group) and arm in r["arms"] and
          "vs_phrasebook" in r["arms"][arm]]
    if not ks:
        return None
    d = np.array([vt[k]["arms"][arm]["vs_phrasebook"]["delta"] for k in ks])
    bs = d[rng.integers(0, len(d), size=(BOOT, len(d)))].mean(1)
    fam = {}
    for k in ks:
        for f, m in vt[k]["arms"][arm]["vs_phrasebook"]["mcnemar"].items():
            x = fam.setdefault(f, [0, 0])
            x[0] += m["gain"]
            x[1] += m["loss"]
    up, down = int((d > 1e-9).sum()), int((d < -1e-9).sum())
    return {"keys": len(ks), "mean_delta": float(d.mean()), "mean_delta_ci": [float(x) for x in np.percentile(bs, [2.5, 97.5])],
            "up": up, "down": down, "tie": len(ks) - up - down, "sign_p": sign_two(up, down),
            "sig_down": [k for k in ks if vt[k]["arms"][arm]["vs_phrasebook"]["p_boot"] < 0.05 and
                         vt[k]["arms"][arm]["vs_phrasebook"]["delta"] < 0],
            "sig_up": [k for k in ks if vt[k]["arms"][arm]["vs_phrasebook"]["p_boot"] < 0.05 and
                       vt[k]["arms"][arm]["vs_phrasebook"]["delta"] > 0],
            "below_floor": [k for k in ks if not vt[k]["arms"][arm]["vs_phrasebook"]["floor_ok"]],
            "judge_pairs": {f: {"gain": g, "loss": l, "p": sign_two(g, l)} for f, (g, l) in fam.items()}}


def fmt_cons(r):
    if not r or r.get("cons") is None:
        return "-"
    lo, hi = r["cons_ci"]
    return "%.2f [%.2f, %.2f]" % (r["cons"], lo, hi)


def fmt_vs(r):
    v = (r or {}).get("vs_phrasebook")
    if not v:
        return "-"
    mc = " ".join("%s +%d/-%d" % (f[0], m["gain"], m["loss"]) for f, m in sorted(v["mcnemar"].items()))
    return "%+.2f [%+.2f, %+.2f] p %.3f%s | %s" % (v["delta"], v["delta_ci"][0], v["delta_ci"][1], v["p_boot"],
                                                  "" if v["floor_ok"] else " BELOW FLOOR", mc)


def merge_vocab(V, work):
    import glob
    for p in sorted(glob.glob(os.path.join(work, "vocab_*.json"))):
        X = load_json(p)
        if X.get("chosen") != V.get("chosen"):
            raise SystemExit("%s chose other judges than vocab.json" % p)
        for t, by in X["chosen"].items():
            for f, v in by.items():
                w = t.split("@")[1]
                if abs(X["validity"][v][w]["real"] - V["validity"][v][w]["real"]) > 1e-12:
                    raise SystemExit("%s measured another real rate for %s" % (p, v))
        routes = set(X["meta"]["settings"])
        own = lambda g: g.startswith("gen:") and g.split("|")[0][4:] in routes
        V["consensus"].update({g: c for g, c in X["consensus"].items() if own(g)})
        V["generated"].update({g: c for g, c in X["generated"].items() if own(g)})
        V.setdefault("shape", {}).update({g: c for g, c in X.get("shape", {}).items() if own(g)})
        V.setdefault("material", {}).setdefault("generated", {}).update(
            {g: c for g, c in X.get("material", {}).get("generated", {}).items() if own(g)})
        print("merged %s: routes %s" % (os.path.basename(p), sorted(routes)), flush=True)


def cmd_report(a):
    rng = np.random.default_rng(BOOT_SEED)
    V = load_json(os.path.join(a.work, "vocab.json"))
    merge_vocab(V, a.work)
    vt = vocab_table(V, rng)
    W = load_json(os.path.join(a.work, "water.json")) if os.path.isfile(os.path.join(a.work, "water.json")) else None
    wt = water_table(W, rng) if W else {}
    rows = render_rows(a.work)
    prompts = prompts_by_src(a.work)
    by = {}
    for r in rows:
        by.setdefault((r["key"], r["model"], r["style"], r["amp"]), []).append(r)
    stat = {}
    for et, mat, grp in KEYS:
        k = kid(et, mat)
        stat[k] = {}
        for arm in RUN:
            src = source(arm, et, mat)
            rr = by.get((k,) + src, [])
            stat[k][arm] = {"model": src[0], "prompt": prompts.get((k, src[0], src[1])), "n": len(rr),
                            "static": int(sum(x["static"] for x in rr)), "t40": float(np.median([x["t40"] for x in rr]))
                            if rr else None, "len": float(np.median([x["len"] for x in rr])) if rr else None,
                            "gen_ms": float(np.median([x["gen_ms"] for x in rr])) if rr else None}
    pool = {arm: {g: pooled(vt, arm, g, rng) for g in ("demo", "unseen", "all")} for arm in RUN if arm != "phrasebook"}
    validity = {}
    for t, by_f in (V.get("chosen") or {}).items():
        validity[t] = {f: {"variant": v, "real": V["validity"][v][t.split("@")[1]]["real"]} for f, v in by_f.items()}
    res = {"vocab": vt, "water": wt, "clips": stat, "pooled": pool, "validity": validity,
           "meta": {"refs": V["meta"]["manifest"], "seeds": V["meta"]["args"]["seeds"], "boot": BOOT}}
    for k in res["vocab"].values():
        for r in k["arms"].values():
            r.pop("hits", None)
    for k in res["water"].values():
        for r in k["arms"].values():
            r.pop("hits", None)
    dump(res, os.path.join(a.work, "report.json"))
    print("IDENTITY (held-out vocab judges, %d seeds): cons = min over judges of min(rate/real, 1) [95%% CI]" %
          res["meta"]["seeds"])
    print("%-30s %-15s %-22s %-22s %-48s %-22s %s" % ("key", "type", "phrasebook", "generic", "generic vs phrasebook",
                                                      "generic_all", "generic_all vs phrasebook"))
    for k, r in vt.items():
        A = r["arms"]
        print("%-30s %-15s %-22s %-22s %-48s %-22s %s" % (
            k, r["type"] + ("*" if r["approx"] else ""), fmt_cons(A.get("phrasebook")), fmt_cons(A.get("generic")),
            fmt_vs(A.get("generic")), fmt_cons(A.get("generic_all")), fmt_vs(A.get("generic_all"))))
        for arm in ("phrasebook_words", "generic_words", "generic_bare"):
            if arm in A and A[arm]["route"] == arm:
                print("%-30s %-15s %-22s %s" % ("", arm, fmt_cons(A[arm]), fmt_vs(A[arm])))
    print()
    print("WATER JUDGE (ESC-50 drops control, 0.6 s)")
    for k, r in wt.items():
        A = r["arms"]
        print("%-30s %-22s %-22s %-48s %-22s %s" % (k + ("*" if r["approx"] else ""), fmt_cons(A.get("phrasebook")),
                                                    fmt_cons(A.get("generic")), fmt_vs(A.get("generic")),
                                                    fmt_cons(A.get("generic_all")), fmt_vs(A.get("generic_all"))))
    print()
    print("POOLED over judged keys (delta of cons vs phrasebook)")
    for arm, gs in pool.items():
        for g, p in gs.items():
            if not p:
                continue
            print("%-17s %-7s keys %2d mean %+.3f [%+.3f, %+.3f] up %d down %d tie %d sign p %.3f  sig down %s  sig up %s  "
                  "below floor %s  judges %s" % (
                      arm, g, p["keys"], p["mean_delta"], p["mean_delta_ci"][0], p["mean_delta_ci"][1], p["up"],
                      p["down"], p["tie"], p["sign_p"], p["sig_down"] or "-", p["sig_up"] or "-", p["below_floor"] or "-",
                      " ".join("%s +%d/-%d p%.3f" % (f, m["gain"], m["loss"], m["p"]) for f, m in
                               sorted(p["judge_pairs"].items()))))
    print()
    print("CLIPS (all %d seeds): model, static-rule flags, median decay to -40 dB, median length" % res["meta"]["seeds"])
    for k, s in stat.items():
        print("%-30s %s" % (k, "  ".join("%s %s st%d t40 %.2f len %.2f" % (
            arm, s[arm]["model"], s[arm]["static"], s[arm]["t40"] or 0, s[arm]["len"] or 0) for arm in MAIN_ARMS
            if arm in RUN)))
    print("saved", os.path.join(a.work, "report.json"))
    return 0


def rank_groups(vt):
    ks = list(vt)
    sa3 = [k for k in ks if source("phrasebook", *k.split("|"))[0] == "sa3"]
    return {"all": ks, "demo": [k for k in ks if vt[k]["group"] == "demo"],
            "unseen": [k for k in ks if vt[k]["group"] == "unseen"],
            "guns": [k for k in ks if vt[k]["type"].startswith("gunshot")],
            "footsteps": [k for k in ks if vt[k]["type"] == "footstep"],
            "sa3_rows": sa3, "woosh_rows": [k for k in ks if k not in sa3]}


def cons_draws(hits, reals, idx):
    out = np.full(idx.shape[0], np.inf)
    for f, hs in hits.items():
        out = np.minimum(out, np.minimum(np.asarray(hs, float)[idx].mean(1) / reals[f], 1.0))
    return out


def rank_table(rows, arms, rng):
    ks = list(rows)
    n = len(next(iter(next(iter(rows[ks[0]].values()))["hits"].values())))
    seeds = {k: rng.integers(0, n, size=(BOOT, n)) for k in ks}
    reals = {k: {f: j["real"] for f, j in rows[k]["phrasebook"]["judges"].items()} for k in ks}
    point, draws = {}, {}
    for arm in arms:
        point[arm] = np.array([cons_of(rows[k][arm]["hits"], reals[k]) for k in ks])
        draws[arm] = np.vstack([cons_draws(rows[k][arm]["hits"], reals[k], seeds[k]) for k in ks])
    return ks, point, draws


def summarize_group(ks, point, draws, members, arms, rng):
    pos = [ks.index(k) for k in members]
    kb = rng.integers(0, len(pos), size=(BOOT, len(pos)))
    sel = np.asarray(pos)[kb]
    cols = np.arange(BOOT)[:, None]
    pooled_b = {arm: draws[arm][sel, cols].mean(1) for arm in arms}
    out = {}
    for arm in arms:
        p = float(point[arm][pos].mean())
        ci = [float(x) for x in np.percentile(pooled_b[arm], [2.5, 97.5])]
        rec = {"mean": p, "ci": ci, "keys": len(pos)}
        for ref in ("phrasebook", "generic"):
            if ref == arm or ref not in pooled_b:
                continue
            d = pooled_b[arm] - pooled_b[ref]
            dp = point[arm][pos] - point[ref][pos]
            rec["vs_" + ref] = {"delta": float(dp.mean()), "ci": [float(x) for x in np.percentile(d, [2.5, 97.5])],
                                "p_boot": min(1.0, 2.0 * min(float(np.mean(d <= 0)), float(np.mean(d >= 0)))),
                                "up": int((dp > 1e-9).sum()), "down": int((dp < -1e-9).sum())}
        out[arm] = rec
    return out


def judge_pairs(rows, members, arm, ref):
    fam = {}
    for k in members:
        a, b = rows[k][arm]["hits"], rows[k][ref]["hits"]
        for f in b:
            ha, hb = np.asarray(a[f], bool), np.asarray(b[f], bool)
            x = fam.setdefault(f, [0, 0])
            x[0] += int((ha & ~hb).sum())
            x[1] += int((~ha & hb).sum())
    return {f: {"gain": g, "loss": l, "p": sign_two(g, l)} for f, (g, l) in sorted(fam.items())}


def fmt_group(rec):
    if not rec:
        return "-"
    return "%.3f [%.2f, %.2f]" % (rec["mean"], rec["ci"][0], rec["ci"][1])


def fmt_delta(rec, ref):
    v = (rec or {}).get("vs_" + ref)
    if not v:
        return "-"
    return "%+.3f [%+.2f, %+.2f] %d/%d" % (v["delta"], v["ci"][0], v["ci"][1], v["up"], v["down"])


def cmd_rank(a):
    rng = np.random.default_rng(BOOT_SEED)
    V = load_json(os.path.join(a.work, "vocab.json"))
    merge_vocab(V, a.work)
    vt = vocab_table(V, rng)
    W = load_json(os.path.join(a.work, "water.json")) if os.path.isfile(os.path.join(a.work, "water.json")) else None
    wt = water_table(W, rng) if W else {}
    arms = [arm for arm in RUN if all(arm in vt[k]["arms"] for k in vt)]
    groups = rank_groups(vt)
    vrows = {k: vt[k]["arms"] for k in vt}
    ks, point, draws = rank_table(vrows, arms, rng)
    res = {"groups": {}, "per_key": {}, "water": {}, "pairs": {}, "floor": {}, "split": {}, "clips": {}, "identity": {}}
    for g, members in groups.items():
        res["groups"][g] = summarize_group(ks, point, draws, members, arms, rng)
    for arm in arms:
        res["pairs"][arm] = {ref: judge_pairs(vrows, ks, arm, ref) for ref in ("phrasebook", "generic") if ref != arm}
        res["floor"][arm] = {ref: [k for k in ks if vrows[k][arm]["cons"] < vrows[k][ref]["cons_ci"][0] - 1e-9]
                             for ref in ("phrasebook", "generic") if ref != arm}
    for k in ks:
        res["per_key"][k] = {arm: {"cons": vrows[k][arm]["cons"], "route": vrows[k][arm]["route"]} for arm in arms}
    warms = [arm for arm in arms if all(arm in wt[k]["arms"] for k in wt)] if wt else []
    if warms:
        wrows = {k: {arm: dict(wt[k]["arms"][arm], judges={f: {"real": r} for f, r in
                                                           wt[k]["arms"]["phrasebook"]["reals"].items()})
                     for arm in warms} for k in wt}
        wks, wpoint, wdraws = rank_table(wrows, warms, rng)
        res["water"] = {"group": summarize_group(wks, wpoint, wdraws, wks, warms, rng),
                        "per_key": {k: {arm: wt[k]["arms"][arm]["cons"] for arm in warms} for k in wks}}
        aks = ks + ["water:" + k for k in wks]
        apoint = {arm: np.concatenate([point[arm], wpoint[arm]]) for arm in warms}
        adraws = {arm: np.vstack([draws[arm], wdraws[arm]]) for arm in warms}
        res["groups"]["all_plus_water"] = summarize_group(aks, apoint, adraws, aks, warms, rng)
    sa3k, wk = groups["sa3_rows"], groups["woosh_rows"]
    sa3_side = {arm: float(np.mean([point[arm][ks.index(k)] for k in sa3k])) for arm in arms}
    woosh_side = {arm: float(np.mean([point[arm][ks.index(k)] for k in wk])) for arm in arms}
    tmpl = [arm for arm in arms if arm not in OLD_ARMS]
    if tmpl:
        bs = max(tmpl, key=lambda x: sa3_side[x])
        bw = max((x for x in tmpl if x.endswith(TT_OFF)), key=lambda x: woosh_side[x])
        pos_s, pos_w = [ks.index(k) for k in sa3k], [ks.index(k) for k in wk]
        sp = np.concatenate([point[bs][pos_s], point[bw][pos_w]])
        order = pos_s + pos_w
        sd = np.vstack([draws[bs][pos_s], draws[bw][pos_w]])
        kb = rng.integers(0, len(order), size=(BOOT, len(order)))
        cols = np.arange(BOOT)[:, None]
        sb = sd[kb, cols].mean(1)
        pb = draws["phrasebook"][np.asarray(order)[kb], cols].mean(1)
        res["split"] = {"sa3": bs, "woosh": bw, "mean": float(sp.mean()),
                        "ci": [float(x) for x in np.percentile(sb, [2.5, 97.5])],
                        "vs_phrasebook": float(sp.mean() - point["phrasebook"][order].mean()),
                        "vs_phrasebook_ci": [float(x) for x in np.percentile(sb - pb, [2.5, 97.5])],
                        "sa3_side": sa3_side, "woosh_side": woosh_side}
    rows = render_rows(a.work)
    by = {}
    for r in rows:
        if r["v"] < a.seeds:
            by.setdefault((r["key"], r["model"], r["style"], r["amp"]), []).append(r)
    for arm in RUN:
        t40 = {"sa3": [], "woosh": []}
        for et, mat, _g in KEYS:
            src = source(arm, et, mat)
            rr = by.get((kid(et, mat),) + src, [])
            if rr:
                t40[src[0]].append(float(np.median([x["t40"] for x in rr])))
        res["clips"][arm] = {m: float(np.median(v)) if v else None for m, v in t40.items()}
    ident = V["generated"].get("_identity", {})
    for k in groups["guns"]:
        res["identity"][k] = {arm: ident.get("gen:%s|%s" % (vrows[k][arm]["route"], k)) for arm in arms}
    dump(res, os.path.join(a.work, "rank.json"))
    order_arms = sorted(arms, key=lambda x: -res["groups"]["all"][x]["mean"])
    print("RANK on %d judged requests, seeds 0-%d: mean of cons (min over judges of min(rate/real, 1)); "
          "[95%% CI, keys and seeds resampled]; deltas paired on key and seed, up/down = keys" % (len(ks), a.seeds - 1))
    hdr = ("arm", "pooled 24", "vs phrasebook", "vs generic", "guns", "footsteps", "unseen", "water judge",
           "24 + water")
    print("%-14s %-22s %-28s %-28s %-22s %-22s %-22s %-22s %s" % hdr)
    for arm in order_arms:
        G = res["groups"]
        print("%-14s %-22s %-28s %-28s %-22s %-22s %-22s %-22s %s" % (
            arm, fmt_group(G["all"][arm]), fmt_delta(G["all"][arm], "phrasebook"), fmt_delta(G["all"][arm], "generic"),
            fmt_group(G["guns"][arm]), fmt_group(G["footsteps"][arm]), fmt_group(G["unseen"][arm]),
            fmt_group(res["water"].get("group", {}).get(arm)) if res["water"] else "-",
            fmt_group(G.get("all_plus_water", {}).get(arm))))
    print()
    print("GROUP DELTAS vs phrasebook / vs generic")
    for g in ("demo", "unseen", "guns", "footsteps", "sa3_rows", "woosh_rows"):
        print(" %s (%d keys)" % (g, len(groups[g])))
        for arm in order_arms:
            r = res["groups"][g][arm]
            print("   %-14s %-22s %-30s %s" % (arm, fmt_group(r), fmt_delta(r, "phrasebook"), fmt_delta(r, "generic")))
    if res["water"]:
        print(" water judge (%d keys)" % len(res["water"]["per_key"]))
        for arm in order_arms:
            r = res["water"]["group"].get(arm)
            print("   %-14s %-22s %-30s %s" % (arm, fmt_group(r), fmt_delta(r, "phrasebook"), fmt_delta(r, "generic")))
    print()
    print("PER KEY cons (seeds 0-%d)" % (a.seeds - 1))
    print("%-30s %s" % ("key", " ".join("%-6s" % x[:6] for x in order_arms)))
    print("%-30s %s" % ("", " ".join("%-6s" % x[6:12] for x in order_arms)))
    for k in ks:
        print("%-30s %s" % (k, " ".join("%-6.2f" % res["per_key"][k][x]["cons"] for x in order_arms)))
    for k, by_arm in (res["water"].get("per_key") or {}).items():
        print("%-30s %s" % ("water:" + k, " ".join("%-6.2f" % by_arm[x] if x in by_arm else "  -   "
                                                   for x in order_arms)))
    print()
    print("JUDGE PAIRS pooled over keys (seeds where one arm hits and the other misses), and keys below the floor "
          "(cons under the reference's lower 95% bound)")
    for arm in order_arms:
        print("  %-14s %s" % (arm, "  ".join("vs %s: %s; below floor %s" % (
            ref, " ".join("%s +%d/-%d" % (f[0], m["gain"], m["loss"]) for f, m in pp.items()),
            res["floor"][arm][ref] or "-") for ref, pp in res["pairs"][arm].items())))
    if res["split"]:
        s = res["split"]
        print()
        print("PER-MODEL SPLIT (best SA3-side arm on its %d keys, best Woosh-side template on its %d keys; picked on "
              "these seeds, so optimistic): sa3 %s + woosh %s = %.3f [%.2f, %.2f], vs phrasebook %+.3f [%+.2f, %+.2f]" % (
                  len(sa3k), len(wk), s["sa3"], s["woosh"], s["mean"], s["ci"][0], s["ci"][1], s["vs_phrasebook"],
                  s["vs_phrasebook_ci"][0], s["vs_phrasebook_ci"][1]))
        print("  sa3 side  " + "  ".join("%s %.2f" % (x, v) for x, v in sorted(s["sa3_side"].items(), key=lambda t: -t[1])))
        print("  woosh side " + "  ".join("%s %.2f" % (x, v) for x, v in sorted(s["woosh_side"].items(),
                                                                                 key=lambda t: -t[1])))
    print()
    print("MEDIAN DECAY to -40 dB (median over keys of the per-key median), s")
    for arm in order_arms:
        c = res["clips"][arm]
        print("  %-14s sa3 %s  woosh %s" % (arm, "%.2f" % c["sa3"] if c["sa3"] is not None else "-",
                                           "%.2f" % c["woosh"] if c["woosh"] is not None else "-"))
    print()
    print("CLAP TOP-1 on gun keys (3 most common)")
    for k, by_arm in res["identity"].items():
        for arm in order_arms:
            print("  %-22s %-14s %s" % (k, arm, by_arm.get(arm)))
    print("saved", os.path.join(a.work, "rank.json"))
    return 0


def h(s):
    return hashlib.md5(s.encode()).hexdigest()


def row_label(et, mat):
    return "%s | %s" % (et, mat)


def cmd_page(a):
    rep = load_json(os.path.join(a.work, "report.json"))
    out = a.out
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(os.path.join(out, "a"))
    rows = render_rows(a.work)
    meta = {(r["key"], r["model"], r["style"], r["amp"], r["v"]): r for r in rows}
    key, info, sections, n_rows, n_clips = {}, {}, {"demo": [], "unseen": []}, 0, 0
    for et, mat, grp in KEYS:
        k = kid(et, mat)
        sid = slug(k)
        opts = {}
        for arm in RUN:
            opts.setdefault(source(arm, et, mat), []).append(arm)
        order = sorted(opts, key=lambda s: h(a.seed + sid + src_name(s)))
        blocks = []
        for i, src in enumerate(order):
            arms = opts[src]
            clips, nums = [], []
            for v in range(a.takes):
                y = np.load(clip_path(a.work, src, et, mat, v))
                r = meta[(k,) + src + (v,)]
                nm = h(a.seed + sid + src_name(src) + str(v))[:12] + ".wav"
                AudioClip(np.ascontiguousarray(y, dtype=np.float32), int(r["sr"])).to_wav(os.path.join(out, "a", nm))
                n_clips += 1
                nums.append(r)
                clips.append("<audio controls preload='none' src='a/%s' aria-label='%s take %d'></audio>" % (
                    nm, LETTERS[i], v + 1))
            letter = LETTERS[i]
            key["%s|%s" % (sid, letter)] = arms
            judged = []
            for arm in arms:
                vr = rep["vocab"].get(k, {}).get("arms", {}).get(arm)
                if vr and vr.get("cons") is not None:
                    judged.append("identity %.2f [%.2f, %.2f]" % (vr["cons"], vr["cons_ci"][0], vr["cons_ci"][1]))
                    break
            for arm in arms:
                wr = rep["water"].get(k, {}).get("arms", {}).get(arm)
                if wr:
                    judged.append("water %.2f [%.2f, %.2f]" % (wr["cons"], wr["cons_ci"][0], wr["cons_ci"][1]))
                    break
            info["%s|%s" % (sid, letter)] = "%s | %s, amp %s | %s | takes: -40 dB after %s s, length %s s | prompt: %s" % (
                ", ".join(arms), MODEL_NAME[src[0]], src[2], "; ".join(judged) or "no validated judge for this sound",
                " / ".join("%.2f" % n["t40"] for n in nums), " / ".join("%.2f" % n["len"] for n in nums),
                nums[0]["prompt"])
            blocks.append("<div class='opt' data-s='%s' data-l='%s'><div class='let'>%s</div><div class='takes'>%s</div>"
                          "<div class='btns'><button data-v='best'>best</button><button data-v='worst'>worst</button>"
                          "</div><div class='rev'></div></div>" % (sid, letter, letter, "".join(clips)))
        sections[grp].append("<section class='snd' data-s='%s'><h3>%s</h3>%s<p class='row'><button class='none' "
                             "data-s='%s'>none sound right</button><span class='st'></span></p></section>" % (
                                 sid, html.escape(row_label(et, mat)), "".join(blocks), sid))
        n_rows += 1
    body = ("<h2>The twelve demo sounds</h2>%s<h2>Sounds another game would send</h2>%s" % (
        "".join(sections["demo"]), "".join(sections["unseen"])))
    payload = base64.b64encode(json.dumps({"key": key, "info": info, "arms": ARM_TEXT}, ensure_ascii=True).encode()).decode()
    page = TEMPLATE.replace("%SECTIONS%", body).replace("%DATA%", payload).replace("%NSOUNDS%", str(n_rows)).replace(
        "%STORE%", "sarge_generic_page_" + a.seed).replace("%TAKES%", str(a.takes))
    with open(os.path.join(out, "index.html"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(page)
    dump({"seed": a.seed, "key": key, "info": info}, os.path.join(os.path.dirname(out), "page_key.json"))
    print("wrote %s (%d rows, %d clips)" % (os.path.join(out, "index.html"), n_rows, n_clips), flush=True)
    return 0


TEMPLATE = """<!doctype html><html lang='en'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Generic Prompts Listening</title><style>
:root{--bg:#f7f7f4;--fg:#1c1c1a;--mute:#66665f;--line:#dcdcd5;--card:#ffffff;--acc:#2456c9;--bad:#b4400c;--ok:#2f7d32}
@media (prefers-color-scheme: dark){:root:not([data-theme='light']){--bg:#151514;--fg:#ececea;--mute:#9c9c96;--line:#34342f;--card:#1d1d1b;--acc:#8fb0ff;--bad:#fb923c;--ok:#7bc47f}}
:root[data-theme='dark']{--bg:#151514;--fg:#ececea;--mute:#9c9c96;--line:#34342f;--card:#1d1d1b;--acc:#8fb0ff;--bad:#fb923c;--ok:#7bc47f}
*{box-sizing:border-box}
html{background:var(--bg);overflow-x:hidden}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,'Segoe UI',sans-serif;margin:0 auto;padding:24px 16px 110px;max-width:1080px;overflow-x:hidden}
h1{font-size:24px;margin:0 0 8px}
h2{font-size:19px;margin:28px 0 4px}
h3{font-size:16px;margin:0 0 6px;overflow-wrap:anywhere}
p{color:var(--mute);max-width:760px;margin:6px 0}
.snd{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin:14px 0}
.opt{display:grid;grid-template-columns:2rem minmax(0,1fr) auto;gap:6px 12px;align-items:center;padding:8px 0;border-top:1px solid var(--line)}
.let{font-weight:700;font-size:17px}
.takes{display:grid;grid-template-columns:repeat(%TAKES%,minmax(0,1fr));gap:6px;min-width:0}
audio{width:100%;min-width:0;height:34px}
.btns{display:flex;gap:6px}
button{font:inherit;font-size:14px;padding:4px 12px;border:1px solid var(--line);background:transparent;color:var(--fg);border-radius:6px;cursor:pointer}
button:hover{border-color:var(--mute)}
button.on{border-color:var(--acc);color:var(--acc);font-weight:600}
button.on[data-v='worst'],button.none.on{border-color:var(--bad);color:var(--bad)}
.rev{grid-column:1/-1;color:var(--acc);font-size:13px;display:none;overflow-wrap:anywhere}
body.revealed .rev{display:block}
.row{display:flex;flex-wrap:wrap;align-items:center;gap:10px;margin:10px 0 0}
.st{font-size:13px;color:var(--mute)}.st.done{color:var(--ok)}
.bar{position:fixed;left:0;right:0;bottom:0;background:var(--card);border-top:1px solid var(--line);padding:10px 16px}
.bar div{max-width:1080px;margin:0 auto;display:flex;flex-wrap:wrap;gap:8px 12px;align-items:center}
#prog{font-size:14px;color:var(--mute);margin-right:auto}
#tally{margin:24px 0}#tally table{border-collapse:collapse;width:100%;font-size:14px}
#tally th,#tally td{text-align:left;border-bottom:1px solid var(--line);padding:5px 8px;vertical-align:top}
#tally td.n{text-align:right;font-variant-numeric:tabular-nums}
.wrap{overflow-x:auto}
@media (max-width:900px){.takes{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media (max-width:640px){.opt{grid-template-columns:2rem minmax(0,1fr)}.btns{grid-column:2}.takes{grid-template-columns:minmax(0,1fr)}h1{font-size:21px}}
</style></head><body>
<h1>Generic prompts: blind listening</h1>
<p>Use headphones. Each row is one request a game sends (event | material), rendered on the models that ship: SA3 medium
for whatever the route calls a gun, Woosh DFlow for everything else. The options are the shipped phrasebook, the
generic flag sets and, on rows SA3 renders, a diagnostic without the 'TrackType: SFX,' prefix; any two that produce the
same audio for a row share one option, so a row has three to six options. Every option plays %TAKES% takes, the first variants the game rotates through. Letters are shuffled per row.</p>
<p>Levels are what the game plays: every clip is the finished clip at the same playback gain. Mark the option that sounds
most like the request and the one that sounds least like it, or press "none sound right". Picks stay in this browser.
Reveal names the flag sets behind each option, the model, the prompt it got and the judge score where a validated judge
exists; Download saves your picks with the key.</p>
%SECTIONS%
<section id='tally'></section>
<div class='bar'><div><span id='prog'></span><button id='reveal'>Reveal</button><button id='dl'>Download picks</button><button id='clear'>Clear picks</button></div></div>
<script>
const D=JSON.parse(atob('%DATA%'));
const SK='%STORE%', N=%NSOUNDS%;
let store={};
try{store=JSON.parse(localStorage.getItem(SK)||'{}')||{}}catch(e){store={}}
let shown=false;
try{shown=localStorage.getItem(SK+'_rev')==='1'}catch(e){}
function save(){try{localStorage.setItem(SK,JSON.stringify(store))}catch(e){}}
function done(s){return !!(s&&(s.none||(s.best&&s.worst)))}
function paint(){
 document.querySelectorAll('.opt').forEach(o=>{const s=store[o.dataset.s]||{};
  o.querySelectorAll('button').forEach(b=>b.classList.toggle('on',s[b.dataset.v]===o.dataset.l));});
 document.querySelectorAll('section.snd').forEach(sec=>{const s=store[sec.dataset.s]||{};
  sec.querySelector('button.none').classList.toggle('on',!!s.none);
  const st=sec.querySelector('.st');
  st.textContent=s.none?'none sound right':(s.best||s.worst)?('best '+(s.best||'-')+', worst '+(s.worst||'-')):'';
  st.classList.toggle('done',done(s));});
 const n=Object.values(store).filter(done).length;
 document.getElementById('prog').textContent=n+' of '+N+' rows rated';
 if(shown)reveal();}
document.querySelectorAll('.opt button').forEach(b=>b.onclick=()=>{const o=b.closest('.opt'),s=o.dataset.s,l=o.dataset.l,v=b.dataset.v;
 const cur=Object.assign({},store[s]||{});cur.none=false;
 if(cur[v]===l){delete cur[v]}else{cur[v]=l;const w=v==='best'?'worst':'best';if(cur[w]===l)delete cur[w];}
 store[s]=cur;save();paint();});
document.querySelectorAll('button.none').forEach(b=>b.onclick=()=>{const s=b.dataset.s;
 store[s]=(store[s]||{}).none?{}:{none:true};save();paint();});
function arms(s,l){return D.key[s+'|'+l]||[]}
function reveal(){
 document.body.classList.add('revealed');
 document.querySelectorAll('.opt').forEach(o=>{o.querySelector('.rev').textContent=D.info[o.dataset.s+'|'+o.dataset.l]||'';});
 const t={};Object.keys(D.arms).forEach(m=>{t[m]={best:0,worst:0}});let none=0;
 Object.entries(store).forEach(([s,v])=>{if(v.none){none++;return}
  arms(s,v.best).forEach(m=>{if(t[m])t[m].best++});arms(s,v.worst).forEach(m=>{if(t[m])t[m].worst++});});
 let h='<h2>Tally</h2><p>An option that stands for several flag sets counts for each of them.</p><div class="wrap"><table><tr><th>flag set</th><th>best</th><th>worst</th><th>what it is</th></tr>';
 Object.entries(D.arms).forEach(([m,d])=>{h+='<tr><td>'+m+'</td><td class="n">'+t[m].best+'</td><td class="n">'+t[m].worst+'</td><td>'+d+'</td></tr>';});
 h+='<tr><td>none sound right</td><td class="n" colspan="2">'+none+'</td><td></td></tr>';
 document.getElementById('tally').innerHTML=h+'</table></div>';
 shown=true;try{localStorage.setItem(SK+'_rev','1')}catch(e){}}
document.getElementById('reveal').onclick=reveal;
document.getElementById('dl').onclick=()=>{const picks={};
 Object.entries(store).forEach(([s,v])=>{picks[s]={best:v.best||null,worst:v.worst||null,none:!!v.none,
  best_flag_sets:v.best?arms(s,v.best):null,worst_flag_sets:v.worst?arms(s,v.worst):null}});
 const x=document.createElement('a');
 x.href=URL.createObjectURL(new Blob([JSON.stringify({page:'generic',saved:new Date().toISOString(),revealed:shown,picks:picks,key:D.key},null,1)],{type:'application/json'}));
 x.download='generic_picks.json';document.body.appendChild(x);x.click();x.remove();};
document.getElementById('clear').onclick=()=>{if(!confirm('Clear every pick on this page?'))return;
 store={};shown=false;save();try{localStorage.removeItem(SK+'_rev')}catch(e){}
 document.body.classList.remove('revealed');document.getElementById('tally').innerHTML='';paint();};
paint();
</script></body></html>"""


def main(argv=None):
    ap = argparse.ArgumentParser(description="the generic runtime path against the tuned phrasebook on the models that "
                                             "ship: render, verify against the sidecar, judge, report, blind page")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("render-woosh", "render-sa3", "verify", "routes", "judge-vocab", "judge-water", "report", "page",
                 "rank"):
        p = sub.add_parser(name)
        p.add_argument("--work", default=WORK)
        p.add_argument("--seeds", type=int, default=SEEDS)
        p.add_argument("--takes", type=int, default=TAKES)
        p.add_argument("--run", choices=sorted(RUNS), default="old",
                       help="old: the shipped flags against the generic flag sets; templates: the phrasebook and the "
                            "generic word list against every universal template with and without the SA3 prefix")
        p.add_argument("--water-dir", default=WATER_DIR,
                       help="tier folders for water_judge.py, which must sit inside the repo")
        if name == "verify":
            p.add_argument("--served", nargs="+", default=[], help="arm=dir of generic_probe.py output, variant 0")
        if name == "judge-vocab":
            p.add_argument("--refs", default=os.path.join("data", "refs_heldout"))
            p.add_argument("--arms", nargs="+", default=None)
            p.add_argument("--tag", default="")
        if name == "render-sa3":
            p.add_argument("--styles", nargs="+", default=None)
        if name == "page":
            p.add_argument("--out", default=PAGE)
            p.add_argument("--seed", default="generic1")
    a = ap.parse_args(argv)
    global RUN, WATER_ARMS
    RUN, WATER_ARMS = RUNS[a.run]
    a.water_dir = os.path.abspath(os.path.join(ROOT, a.water_dir))
    return {"render-woosh": cmd_render_woosh, "render-sa3": cmd_render_sa3, "verify": cmd_verify, "routes": cmd_routes,
            "judge-vocab": cmd_judge_vocab, "judge-water": cmd_judge_water, "report": cmd_report,
            "page": cmd_page, "rank": cmd_rank}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())

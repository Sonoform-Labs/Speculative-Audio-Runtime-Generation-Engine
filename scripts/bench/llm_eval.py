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
import re
import shutil
import time

import numpy as np

import generic_eval as ge
from sarge import scene as scenes
from sarge import writer as wr
from sarge.generators.routed import side_for
from sarge.prompts import build, compose, template, template_slots
from sarge.types import AudioClip, CacheKey
from sarge.world import bucket

ROOT = ge.ROOT
SCRIPTS = ge.SCRIPTS
GATE = os.path.join(ROOT, "data", "_gate", "llm")
SCENES = os.path.join(GATE, "scenes40.json")
WORK = os.path.join(GATE, "_work")
WATER_DIR = os.path.join(GATE, "_water")
PAGE = os.path.join(GATE, "page")
BASE_ARMS = ("phrasebook", "caption", "llm")
VARIANTS = ("llm_nosound", "llm_named")
ALL_ARMS = BASE_ARMS + VARIANTS
ARMS = BASE_ARMS
CAPTION = "template:caption+tt"
FIRST, SEEDS, TAKES = 40, 20, 4
AMP = "full"
ROUTE = "families"
BENCH_TYPE = {"pistol_shot": "gunshot_pistol", "rifle_shot": "gunshot_rifle", "step_stone": "footstep",
              "step_gravel": "footstep", "step_wood": "footstep", "step_metal": "footstep", "step_grass": "footstep",
              "step_water": "footstep", "iron_boots_step": "footstep", "bottle_shot_pistol": "glass_break",
              "window_rifle": "glass_break", "punch": "punch", "car_horn": "horn", "door_creak": "door_open",
              "dog_bark": "dog_bark", "sword_clash": "sword", "keyboard_typing": "ui_click", "explosion": "explosion",
              "snow_step": "footstep", "glass_shatter": "glass_break", "car_door_slam": "door_slam"}
BENCH_APPROX = {"bottle_shot_pistol": "glass-break references for a bullet through a bottle",
                "keyboard_typing": "mouse, button and keyboard clicks stand in for typing",
                "explosion": "fireworks stand in for an exploding barrel",
                "car_door_slam": "house door slams stand in for a car door",
                "glass_shatter": "window and glass breaks stand in for a wine glass"}
BENCH_WATER = {"step_water": None, "drip_puddle": None, "rain_tin_roof": "water drops stand in for rain on a tin roof"}
BENCH_NO_JUDGE = {"pistol_reload": "gun mechanism references too thin on the held-out split",
                  "rifle_reload": "gun mechanism references too thin on the held-out split",
                  "ball_grass": "no reference class for a ball bounce", "ball_metal": "no reference class for a ball bounce",
                  "ball_stone": "no reference class for a ball bounce", "ball_wood": "no reference class for a ball bounce",
                  "crate_fall": "no reference class for a crate drop", "barrel_fall": "no reference class for a barrel drop",
                  "campfire": "crackling fire is a negatives-only set", "lightsaber_swing": "no reference class",
                  "grapple_fire": "no reference class", "coin_pickup": "shell-drop references too thin",
                  "monster_growl": "no reference class", "bow_release": "no reference class",
                  "laser_zap": "no reference class", "magic_missile": "no reference class", "ice_crack": "no reference class"}
ARM_TEXT = {"phrasebook": "the shipped phrasebook (--prompts tuned, the default): sarge.prompts.build, with the "
                          "--world physics template on the six physics scenes",
            "caption": "the universal template --prompts template:caption+tt: 'A single {intensity} {material} "
                       "{event} sound, recorded up close.' with 'TrackType: SFX, ' on SA3 and the contact words of "
                       "--world on physics scenes",
            "llm": "the prompt writer (--writer llama, Gemma 4 E2B-it QAT on the CPU): three fields from the key, "
                   "composed in the SA3 or the Woosh form; a rejected answer keeps the phrasebook",
            "llm_nosound": "diagnostic, no flag: the writer's source and action without its sound field, same tails",
            "llm_named": "diagnostic, no flag: the writer's prompt with the request's material and event words put "
                         "in front whenever the writer's source line lacks one of them"}
TIERS = {arm: "%s_%s" % ("abcde"[i], arm) for i, arm in enumerate(ALL_ARMS)}
SET_TEXT = {"bench": "the 40 scene-bearing bench scenarios", "generic": "the 35 plain requests of the generic run"}
GROUP_TEXT = {"all": "every judged scenario", "demo": "demo scenarios, both sets", "unseen": "unseen scenarios, both sets",
              "bench": "scene-bearing bench scenarios", "bench_demo": "bench, demo", "bench_unseen": "bench, unseen",
              "generic": "plain requests of the generic run", "generic_demo": "generic run, demo",
              "generic_unseen": "generic run, unseen", "guns": "gunshots", "footsteps": "footsteps",
              "sa3_rows": "rows SA3 medium renders", "woosh_rows": "rows Woosh renders", "water": "water judge rows",
              "all_plus_water": "every judged scenario plus the water rows"}


def sid_of(kind, name):
    return ("b_" if kind == "bench" else "g_") + re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def stem_of(sid):
    return re.sub(r"[^a-z]", "", sid) + "-s"


def build_scenarios():
    out = []
    with open(SCENES, encoding="utf-8") as fh:
        bench = json.load(fh)
    for r in bench:
        phys = r.get("phys")
        b = bucket(phys, r["event_type"]) if isinstance(phys, dict) else ""
        sc = scenes.quantise(r["scene"], phys if isinstance(phys, dict) else None, r["event_type"])
        k = CacheKey(r["event_type"], r["material"], r["intensity"], r["context"], phys=b, scene=sc)
        sid = sid_of("bench", r["id"])
        out.append({"sid": sid, "set": "bench", "group": "demo" if r["group"] == "demo" else "unseen",
                    "name": r["id"], "key": wr.key_fields(k), "type": BENCH_TYPE.get(r["id"]),
                    "approx": BENCH_APPROX.get(r["id"]), "no_judge": BENCH_NO_JUDGE.get(r["id"]),
                    "water": stem_of(sid) if r["id"] in BENCH_WATER else None,
                    "water_approx": BENCH_WATER.get(r["id"]), "note": r.get("note", "")})
    for et, mat, g in ge.KEY_SETS["old"]:
        k = CacheKey(et, mat, ge.INTENSITY, ge.CONTEXT)
        kk = ge.kid(et, mat)
        sid = sid_of("generic", kk)
        out.append({"sid": sid, "set": "generic", "group": g, "name": kk, "key": wr.key_fields(k),
                    "type": ge.VOCAB.get(kk), "approx": ge.APPROX.get(kk), "no_judge": ge.NO_JUDGE.get(kk),
                    "water": stem_of(sid) if kk in ge.WATER else None,
                    "water_approx": ge.APPROX.get(kk) if kk in ge.WATER else None, "note": ""})
    for r in out:
        k = wr.key_of(r["key"])
        r["model"] = "sa3" if side_for(k, ROUTE) == "guns" else "woosh"
        r["text"] = wr.key_text(k)
    sids = [r["sid"] for r in out]
    stems = [r["water"] for r in out if r["water"]]
    if len(set(sids)) != len(sids) or len(set(stems)) != len(stems):
        raise SystemExit("scenario ids or water stems collide")
    if not all(re.fullmatch(r"[a-z]+-[a-z]+", s) for s in stems):
        raise SystemExit("water stems must be letters only")
    return out


def render_key(r):
    k = r["key"]
    return CacheKey(k["event_type"], k["material"], k["intensity"], k["context"], k["phys"])


def phrase_prompt(k, model):
    if model == "sa3":
        return build(k.event_type, k.material, k.intensity, k.context)
    return compose("tuned", k, "")


def caption_prompt(k, model):
    return template(CAPTION, k.event_type, k.material, k.intensity, k.context, k.phys, model)


def cmp_pairs(arms):
    out = [(x, "phrasebook") for x in arms if x != "phrasebook"]
    out += [(x, "caption") for x in arms if x not in ("phrasebook", "caption")]
    out += [(x, "llm") for x in arms if x in VARIANTS]
    return tuple(out)


def _near(w, words):
    return any(w == x or (len(w) >= 5 and w[:5] == x[:5]) for x in words)


def variant_prompt(arm, slots, k, model):
    src, act, snd = (wr._clean(slots.get(x)) for x in wr.SLOTS)
    contact, room = wr.tail_words(k)
    head = ""
    if arm == "llm_nosound":
        snd = ""
    elif arm == "llm_named":
        mf = template_slots(k.event_type, k.material, k.intensity, k.context, k.phys)["mf"]
        have = wr._words(src)
        head = mf if any(not _near(w, have) for w in wr._words(mf)) else ""
    if model == "sa3":
        first = head or src
        parts = (first[:1].upper() + first[1:], src if head else "", act, snd, contact, wr.SA3_FRAME, room)
        return wr.SA3_PREFIX + ", ".join(p for p in parts if p) + "."
    return ", ".join(p for p in (head, src, act, snd, contact, room) if p)


def cmd_variants(a):
    S = load(scen_path(a.work))
    for r in S["rows"]:
        k = wr.key_of(r["key"])
        for arm in VARIANTS:
            r["prompts"][arm] = (variant_prompt(arm, r["writer"]["slots"], k, r["model"]) if r["writer"]["ok"]
                                 else r["prompts"]["phrasebook"])
    dump(S, scen_path(a.work))
    for r in S["rows"]:
        print("%-30s %s" % (r["sid"], " || ".join(r["prompts"][x] for x in ("llm",) + VARIANTS)))
    return 0


def clip_id(model, rk, text):
    return hashlib.sha1(("%s\n%s\n%s\n%s" % (model, rk, text, AMP)).encode("utf-8")).hexdigest()[:16]


def dump(obj, p):
    ge.dump(obj, p)


def load(p):
    return ge.load_json(p)


def scen_path(work):
    return os.path.join(work, "scenarios.json")


def cmd_write(a):
    rows = build_scenarios()
    os.makedirs(a.work, exist_ok=True)
    store_dir = os.path.join(a.work, "writer")
    pw = wr.build("llama", model=a.model, port=a.port, threads=a.threads, store=store_dir,
                  log_path=os.path.join(a.work, "writer_server.log"))
    pw.setup()
    t0 = time.perf_counter()
    todo = [r for r in rows if str(wr.key_of(r["key"])) not in pw.store]
    load_s = None
    if todo:
        pw.backend.start()
        load_s = time.perf_counter() - t0
        print("writer %s ready in %.1f s, affinity %s, id %s" % (
            os.path.basename(pw.backend.model), load_s, hex(pw.backend.seen_affinity or 0), pw.id[:12]), flush=True)
    for r in todo:
        k = wr.key_of(r["key"])
        rec = pw.write(k)
        pw.store.put(str(k), rec)
        print("%-30s %6.0f ms %-5s %s" % (r["sid"], rec["ms"], "FAIL" if rec.get("failed") else
                                         ("fix" if rec.get("repaired") else "ok"),
                                         rec.get(r["model"]) or rec.get("after") or rec.get("flags")), flush=True)
    if todo:
        pw.store.flush()
        pw.backend.close()
    for r in rows:
        k = wr.key_of(r["key"])
        rec = pw.store.get(str(k))
        ok = wr.ok_record(rec)
        r["prompts"] = {"phrasebook": phrase_prompt(k, r["model"]), "caption": caption_prompt(k, r["model"]),
                        "llm": rec[r["model"]] if ok else phrase_prompt(k, r["model"])}
        r["writer"] = {"ok": ok, "repaired": bool(rec and rec.get("repaired")), "flags": (rec or {}).get("flags"),
                       "slots": (rec or {}).get("slots"), "orig": (rec or {}).get("orig"), "raw": (rec or {}).get("raw"),
                       "ms": (rec or {}).get("ms"), "woosh": (rec or {}).get("woosh"), "sa3": (rec or {}).get("sa3")}
    ms = sorted(r["writer"]["ms"] for r in rows if r["writer"]["ms"] is not None)
    meta = {"writer_id": pw.id, "store": pw.store.path, "model": getattr(pw.backend, "model", None),
            "load_s": load_s, "written_now": len(todo), "route": ROUTE, "amp": AMP,
            "ms_p50": float(np.median(ms)) if ms else None,
            "failed": [r["sid"] for r in rows if not r["writer"]["ok"]],
            "repaired": [r["sid"] for r in rows if r["writer"]["repaired"]]}
    dump({"meta": meta, "rows": rows}, scen_path(a.work))
    print("saved %s: %d scenarios, %d written now, %d rejected, %d repaired" % (
        scen_path(a.work), len(rows), len(todo), len(meta["failed"]), len(meta["repaired"])), flush=True)
    return 0


def scenarios(work, only=None):
    rows = load(scen_path(work))["rows"]
    if only:
        rows = [r for r in rows if r["sid"] in only]
    return rows


def jobs(rows, model):
    out = {}
    for r in rows:
        if r["model"] != model:
            continue
        rk = render_key(r)
        for arm in ARMS:
            text = r["prompts"][arm]
            cid = clip_id(model, str(rk), text)
            j = out.setdefault(cid, {"cid": cid, "model": model, "rkey": rk, "text": text, "uses": []})
            j["uses"].append([r["sid"], arm])
    return [out[c] for c in sorted(out, key=lambda c: (str(out[c]["rkey"]), out[c]["text"]))]


def npy_path(work, cid, v):
    return os.path.join(work, "clips", cid, "%02d.npy" % v)


def cut_file(work, cid, v):
    return os.path.join(work, "cuts", cid, "%02d.wav" % v)


def save(work, cid, v, y, sr, first, takes):
    y = np.ascontiguousarray(y, dtype=np.float32)
    if v - first < takes:
        p = npy_path(work, cid, v)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        np.save(p, y)
    AudioClip(ge.judge_cut(y, sr), ge.JSR).to_wav(cut_file(work, cid, v))
    return dict(ge.facts(y, sr), sha1=hashlib.sha1(y.tobytes()).hexdigest())


def seeds(a):
    return range(a.first_seed, a.first_seed + a.seeds)


def cmd_render_woosh(a):
    sys.path.insert(1, SCRIPTS)
    import woosh_server as ws
    opts = dict(ws.OPTIONS, **ws.ADOPTED["dflow"])
    frames = ws.parse_frames(None)
    t0 = time.perf_counter()
    gen = ws.make_gen("dflow", frames["other"], 4, "dopri5", **opts)
    ws.prepare(gen, frames)
    print("woosh dflow ready in %.1f s, %s" % (time.perf_counter() - t0, opts), flush=True)
    out = []
    for j in jobs(scenarios(a.work, a.only), "woosh"):
        ms = []
        for v in seeds(a):
            t = time.perf_counter()
            raw = ws.render_key(gen, j["rkey"], v, frames, "tuned", prompt=j["text"])
            ms.append((time.perf_counter() - t) * 1000.0)
            y = ws.finish_key(raw, j["rkey"], AMP, True)
            out.append(dict(save(a.work, j["cid"], v, y, ws.SR, a.first_seed, a.takes), cid=j["cid"], model="woosh",
                            rkey=str(j["rkey"]), prompt=j["text"], v=v, sr=ws.SR, gen_ms=round(ms[-1], 2),
                            uses=j["uses"]))
        print("woosh %-60s median %.1f ms  %s  <- %s" % (str(j["rkey"])[:60], float(np.median(ms)), j["text"],
                                                         ",".join("%s:%s" % tuple(u) for u in j["uses"])), flush=True)
    dump({"meta": {"model": "woosh-dflow", "options": opts, "frames": frames, "first_seed": a.first_seed,
                   "seeds": a.seeds, "takes": a.takes, "graph_error": gen.stats.get("graph_error"),
                   "seconds": time.perf_counter() - t0}, "rows": out}, os.path.join(a.work, "render_woosh%s.json" % tag(a)))
    print("saved %d rows" % len(out), flush=True)
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
    g.amp, g.amp_mode = True, AMP
    print("sa3 medium ready in %.1f s" % (time.perf_counter() - t0), flush=True)
    out = []
    for j in jobs(scenarios(a.work, a.only), "sa3"):
        ms, retries = [], 0
        for v in seeds(a):
            r0 = int(g.static_stats.get("static_retries", 0))
            t = time.perf_counter()
            clip = g.generate(j["rkey"], variant=v, prompt=j["text"])
            ms.append((time.perf_counter() - t) * 1000.0)
            rr = int(g.static_stats.get("static_retries", 0)) - r0
            retries += rr
            out.append(dict(save(a.work, j["cid"], v, clip.samples, clip.sample_rate, a.first_seed, a.takes),
                            cid=j["cid"], model="sa3", rkey=str(j["rkey"]), prompt=j["text"], v=v,
                            sr=int(clip.sample_rate), gen_ms=round(ms[-1], 2), static_retries=rr, uses=j["uses"]))
        print("sa3   %-40s median %.1f ms  retries %d  %s  <- %s" % (
            str(j["rkey"])[:40], float(np.median(ms)), retries, j["text"],
            ",".join("%s:%s" % tuple(u) for u in j["uses"])), flush=True)
    dump({"meta": {"model": "sa3-medium", "kwargs": kw, "first_seed": a.first_seed, "seeds": a.seeds, "takes": a.takes,
                   "fast_stats": g.fast_stats(), "seconds": time.perf_counter() - t0}, "rows": out},
         os.path.join(a.work, "render_sa3%s.json" % tag(a)))
    print("saved %d rows" % len(out), flush=True)
    return 0


def tag(a):
    return "_" + a.tag if getattr(a, "tag", "") else ""


def render_rows(work):
    import glob
    rows = []
    for p in sorted(glob.glob(os.path.join(work, "render_*.json"))):
        rows += load(p)["rows"]
    return rows


def arm_clip(r, arm):
    return clip_id(r["model"], str(render_key(r)), r["prompts"][arm])


def judged(rows):
    return [r for r in rows if r["type"]]


def cmd_routes(a):
    rows = scenarios(a.work)
    for arm in ARMS:
        d = os.path.join(a.work, "routes", arm)
        if os.path.isdir(d):
            shutil.rmtree(d)
        os.makedirs(d)
        for r in judged(rows):
            cid = arm_clip(r, arm)
            for v in seeds(a):
                ge.link(cut_file(a.work, cid, v), os.path.join(d, "%s-s_%02d.wav" % (r["sid"], v - a.first_seed)))
        dump({"clusters": [], "sources": [], "steps": None,
              "prompts": {"%s|s" % r["sid"]: r["prompts"][arm] for r in judged(rows)},
              "note": "%s; served path rendered in process (woosh_server.render_key + finish_key with --amp full and "
                      "stereo, StableAudio3Generator with run_sidecar HYBRID_GUNS), route %s; judged clip = mono 0.6 s "
                      "_extract window at normalize_loudness target_for('normal'), 44.1 kHz; variants %d-%d, numbered "
                      "from 00" % (ARM_TEXT[arm], ROUTE, a.first_seed, a.first_seed + a.seeds - 1)},
             os.path.join(d, "route.json"))
    wd = a.water_dir
    if os.path.isdir(wd):
        shutil.rmtree(wd)
    for arm in ARMS:
        d = os.path.join(wd, TIERS[arm])
        os.makedirs(d)
        for r in rows:
            if not r["water"]:
                continue
            cid = arm_clip(r, arm)
            for v in seeds(a):
                ge.link(cut_file(a.work, cid, v), os.path.join(d, "%s_%02d.wav" % (r["water"], v - a.first_seed)))
    os.makedirs(os.path.join(a.work, "water_empty"), exist_ok=True)
    print("routes: %d judged scenarios per arm, water tiers %s" % (len(judged(rows)), sorted(os.listdir(wd))), flush=True)
    return 0


def cmd_judge_vocab(a):
    import vocab_eval as ve
    ve.KEYS = [(r["sid"], "s", r["type"]) for r in judged(scenarios(a.work))]
    argv = ["--refs", a.refs, "--settings", "none", "--seeds", str(a.seeds), "--out",
            os.path.join(a.work, "vocab%s.json" % tag(a)), "--listen", os.path.join(a.work, "vocab_listen%s.wav" % tag(a))]
    for arm in ARMS:
        argv += ["--route", "%s=%s" % (arm, os.path.join(a.work, "routes", arm))]
    return ve.main(argv)


def cmd_judge_water(a):
    import water_judge as wj
    return wj.main(["--ctl-lengths", "0.6", "--probe-json", os.path.join(a.work, "no_probe.json"), "--probe-dir",
                    os.path.join(a.work, "water_empty"), "--tier-dir", a.water_dir, "--out",
                    os.path.join(a.work, "water%s.json" % tag(a))])


def vocab_rows(V, rows):
    cons, gen = V["consensus"], V["generated"]
    shape = V.get("shape", {})
    ident = gen.get("_identity", {})
    out = {}
    for r in judged(rows):
        arms = {}
        for arm in ARMS:
            g = "gen:%s|%s|s" % (arm, r["sid"])
            c = cons.get(g)
            if not c or not c.get("judges"):
                break
            arms[arm] = {"cons": c.get("cons"), "cons_ci": c.get("cons_ci"),
                         "judges": {f: {"variant": j["variant"], "rate": j["rate"], "real": j["real"]}
                                    for f, j in c["judges"].items()},
                         "hits": {f: gen[g][j["variant"]]["hits"] for f, j in c["judges"].items()},
                         "shape": shape.get(g), "identity": ident.get(g)}
        if len(arms) != len(ARMS):
            continue
        base = arms["phrasebook"]
        out[r["sid"]] = {"sid": r["sid"], "set": r["set"], "group": r["group"], "type": r["type"], "approx": r["approx"],
                         "model": r["model"], "reals": {f: j["real"] for f, j in base["judges"].items()},
                         "variants": {f: j["variant"] for f, j in base["judges"].items()},
                         "arms": {arm: arms[arm]["hits"] for arm in ARMS},
                         "cons_ci": {arm: arms[arm]["cons_ci"] for arm in ARMS},
                         "shape": {arm: arms[arm]["shape"] for arm in ARMS},
                         "identity": {arm: arms[arm]["identity"] for arm in ARMS}}
    return out


def water_rows(W, rows):
    ch = W["selection"]["chosen"].get("0.6") or {}
    val = W["validity"]
    gen = W["generated"]
    out = {}
    for r in rows:
        if not r["water"]:
            continue
        wk = "%s|%s" % tuple(r["water"].split("-"))
        arms, reals, variants = {}, {}, {}
        for arm in ARMS:
            tg = "gen:tier|" + TIERS[arm]
            hits = {}
            for f, v in ch.items():
                rec = gen.get(v, {}).get(tg)
                if not rec:
                    continue
                hs = sorted((int(p[0].split("#")[1]), int(p[1])) for p in rec["pairs"] if p[0].split("#")[0] == wk)
                hits[f] = [x for _s, x in hs]
                reals[f] = val[v]["0.6"]["real"]
                variants[f] = v
            if hits:
                arms[arm] = hits
        if len(arms) != len(ARMS):
            continue
        out[r["sid"]] = {"sid": r["sid"], "set": r["set"], "group": "water", "kind": r["group"], "type": "water",
                         "approx": r["water_approx"], "model": r["model"], "reals": reals, "variants": variants,
                         "arms": arms, "cons_ci": {}}
    rng = np.random.default_rng(ge.BOOT_SEED + 1)
    for k, w in out.items():
        for arm in ARMS:
            n = len(next(iter(w["arms"][arm].values())))
            idx = rng.integers(0, n, size=(ge.BOOT, n))
            cb = ge.cons_draws(w["arms"][arm], w["reals"], idx)
            w["cons_ci"][arm] = [float(x) for x in np.percentile(cb, [2.5, 97.5])]
    return out


def groups_of(J):
    ks = list(J)
    g = {"all": ks, "demo": [k for k in ks if J[k]["group"] == "demo"],
         "unseen": [k for k in ks if J[k]["group"] == "unseen"]}
    for s in ("bench", "generic"):
        g[s] = [k for k in ks if J[k]["set"] == s]
        for grp in ("demo", "unseen"):
            g["%s_%s" % (s, grp)] = [k for k in ks if J[k]["set"] == s and J[k]["group"] == grp]
    g["guns"] = [k for k in ks if J[k]["type"].startswith("gunshot")]
    g["footsteps"] = [k for k in ks if J[k]["type"] == "footstep"]
    g["sa3_rows"] = [k for k in ks if J[k]["model"] == "sa3"]
    g["woosh_rows"] = [k for k in ks if J[k]["model"] == "woosh"]
    return {n: m for n, m in g.items() if m}


def fmt_ci(v, ci, sign=True):
    return ge.fmt_ci(v, ci, sign)


def cons(J, k, arm):
    return ge.cons_of(J[k]["arms"][arm], J[k]["reals"])


def clip_facts(rows, rr):
    by = {}
    for x in rr:
        by.setdefault(x["cid"], []).append(x)
    out = {}
    for r in rows:
        out[r["sid"]] = {}
        for arm in ARMS:
            xs = by.get(arm_clip(r, arm), [])
            out[r["sid"]][arm] = {"cid": arm_clip(r, arm), "n": len(xs),
                                  "t40": float(np.median([x["t40"] for x in xs])) if xs else None,
                                  "len": float(np.median([x["len"] for x in xs])) if xs else None,
                                  "static": int(sum(x["static"] for x in xs)),
                                  "gen_ms": float(np.median([x["gen_ms"] for x in xs])) if xs else None}
    return out


def fmt_shape(sh):
    return ge.fmt_shape(sh)


def fmt_ident(x):
    return ge.fmt_ident(x)


def water_merged(work):
    import glob
    ps = sorted(glob.glob(os.path.join(work, "water*.json")))
    if not ps:
        return None
    W = load(ps[0])
    for p in ps[1:]:
        X = load(p)
        if X["selection"]["chosen"].get("0.6") != W["selection"]["chosen"].get("0.6"):
            raise SystemExit("%s chose other water judges" % p)
        for v, by in X["generated"].items():
            W["generated"].setdefault(v, {}).update(by)
    return W


def cmd_report(a):
    CMP = cmp_pairs(ARMS)
    ge.RUN, ge.CMP = ARMS, CMP
    rng = np.random.default_rng(ge.BOOT_SEED)
    S = load(scen_path(a.work))
    rows = S["rows"]
    by_sid = {r["sid"]: r for r in rows}
    V = load(os.path.join(a.work, "vocab.json"))
    ge.merge_vocab(V, a.work)
    J = vocab_rows(V, rows)
    WM = water_merged(a.work)
    W = water_rows(WM, rows) if WM else {}
    facts = clip_facts(rows, render_rows(a.work))
    per = {}
    for k, r in J.items():
        per[k] = {"set": r["set"], "group": r["group"], "type": r["type"], "approx": r["approx"], "model": r["model"],
                  "cons": {arm: cons(J, k, arm) for arm in ARMS}, "cons_ci": r["cons_ci"],
                  "cmp": {"%s vs %s" % (x, y): ge.paired_key(r, x, y, rng) for x, y in CMP},
                  "identity": r["identity"], "shape": r["shape"], "judges": r["variants"]}
    water = {}
    for k, r in W.items():
        water[k] = {"set": r["set"], "kind": r["kind"], "approx": r["approx"], "model": r["model"],
                    "cons": {arm: cons(W, k, arm) for arm in ARMS}, "cons_ci": r["cons_ci"],
                    "cmp": {"%s vs %s" % (x, y): ge.paired_key(r, x, y, rng) for x, y in CMP}, "judges": r["variants"]}
    G = {n: ge.group_stats(m, J, rng) for n, m in groups_of(J).items()}
    if W:
        G["water"] = ge.group_stats(list(W), W, rng)
        for s in ("bench", "generic"):
            m = [k for k in W if W[k]["set"] == s]
            if m:
                G["water_" + s] = ge.group_stats(m, W, rng)
        AW = dict(J, **{"water:" + k: r for k, r in W.items()})
        G["all_plus_water"] = ge.group_stats(list(AW), AW, rng)
        UW = dict({k: J[k] for k in J if J[k]["group"] == "unseen"},
                  **{"water:" + k: r for k, r in W.items() if r["kind"] == "unseen"})
        G["unseen_plus_water"] = ge.group_stats(list(UW), UW, rng)
        DW = dict({k: J[k] for k in J if J[k]["group"] == "demo"},
                  **{"water:" + k: r for k, r in W.items() if r["kind"] == "demo"})
        G["demo_plus_water"] = ge.group_stats(list(DW), DW, rng)
    res = {"meta": dict(S["meta"], first_seed=a.first_seed, seeds=a.seeds, boot=ge.BOOT, refs=V["meta"].get("manifest")),
           "per": per, "water": water, "groups": G, "clips": facts,
           "prompts": {r["sid"]: {"set": r["set"], "group": r["group"], "name": r["name"], "model": r["model"],
                                  "text": r["text"], "prompts": r["prompts"], "writer": r["writer"],
                                  "type": r["type"], "no_judge": r["no_judge"]} for r in rows},
           "validity": V.get("chosen")}
    dump(res, os.path.join(a.work, "report%s.json" % tag(a)))
    lines = []
    say = lines.append
    say("LLM PROMPT WRITER ON SOUND: %d scenarios (%s; %s), variants %d-%d, route %s, --amp %s, stereo" % (
        len(rows), SET_TEXT["bench"], SET_TEXT["generic"], a.first_seed, a.first_seed + a.seeds - 1, ROUTE, AMP))
    say("writer %s (id %s), rejected %s, repaired %s" % (os.path.basename(str(S["meta"].get("model"))),
                                                         str(S["meta"].get("writer_id"))[:12],
                                                         S["meta"]["failed"] or "none", S["meta"]["repaired"] or "none"))
    say("cons = min over valid held-out judges of min(rate / real, 1); deltas paired on seed, 95%% CI from %d draws; "
        "McNemar a/p/c = AST/PANNs/CLAP seeds gained/lost" % ge.BOOT)
    say("")
    say("PER SCENARIO (identity judges)")
    say("%-34s %-16s %-6s %-6s %-18s %-18s %-18s" % ("scenario", "type", "group", "model", "phrasebook", "caption", "llm"))
    for k, r in per.items():
        say("%-34s %-16s %-6s %-6s %s" % (k, r["type"] + ("*" if r["approx"] else ""), r["group"], r["model"], " ".join(
            "%-18s" % fmt_ci(r["cons"][arm], r["cons_ci"][arm], False) for arm in ARMS)))
        for x, y in CMP:
            c = r["cmp"]["%s vs %s" % (x, y)]
            say("%36s %-24s %-26s p %.3f  %s" % ("", "%s vs %s" % (x, y), fmt_ci(c["delta"], c["ci"]), c["p_boot"],
                                                ge.fmt_mc(c["mcnemar"])))
    say("")
    say("WATER JUDGE")
    for k, r in water.items():
        say("%-34s %-6s %s" % (k + ("*" if r["approx"] else ""), r["kind"], " ".join(
            "%-18s" % fmt_ci(r["cons"][arm], r["cons_ci"][arm], False) for arm in ARMS)))
        for x, y in CMP:
            c = r["cmp"]["%s vs %s" % (x, y)]
            say("%36s %-24s %-26s p %.3f  %s" % ("", "%s vs %s" % (x, y), fmt_ci(c["delta"], c["ci"]), c["p_boot"],
                                                ge.fmt_mc(c["mcnemar"])))
    say("")
    say("POOLED (mean of cons over scenarios [95% CI, scenarios and seeds resampled])")
    for g, s in G.items():
        say(" %s, %d rows: %s" % (GROUP_TEXT.get(g, g), len(s["keys"]), "  ".join(
            "%s %s" % (arm, fmt_ci(s["arms"][arm]["mean"], s["arms"][arm]["ci"], False)) for arm in ARMS)))
        for name, c in s["cmp"].items():
            say("    %-24s %-26s p %.3f  rows up/down %d/%d (sign p %.3f)  judges %s" % (
                name, fmt_ci(c["delta"], c["ci"]), c["p_boot"], c["up"], c["down"], c["sign_p"], " ".join(
                    "%s +%d/-%d p%.3f" % (f, x["gain"], x["loss"], x["p"]) for f, x in c["judges"].items())))
    for arm in ARMS:
        others = [o for o in ARMS if o != arm]
        hit = [k for k, r in per.items() if r["cons"][arm] < 0.5 or any(
            r["cons"][o] - r["cons"][arm] >= 0.2 for o in others)]
        if not hit:
            continue
        say("")
        say("WHERE %s FAILS (cons under 0.5, or 0.2 or more under another arm): CLAP top-1, shape medians, prompt" % arm)
        for k in hit:
            r, cl = per[k], facts[k]
            say(" %s %s (%s): %s" % (k, r["type"], r["model"], "  ".join("%s %.2f" % (o, r["cons"][o]) for o in ARMS)))
            for o in ARMS:
                say("    %-10s CLAP %s" % (o, fmt_ident(r["identity"][o])))
                say("    %-10s %s; -40 dB after %.2f s, length %.2f s" % ("", fmt_shape(r["shape"][o]), cl[o]["t40"] or 0,
                                                                         cl[o]["len"] or 0))
                say("    %-10s prompt: %s" % ("", by_sid[k]["prompts"][o]))
    wfail = [k for k, r in water.items() if any(r["cons"][o] < 0.5 for o in ARMS)]
    if wfail:
        say("")
        say("WATER ROWS WITH AN ARM UNDER 0.5")
        for k in wfail:
            say(" %s: %s" % (k, "  ".join("%s %.2f" % (o, water[k]["cons"][o]) for o in ARMS)))
            for o in ARMS:
                say("    %-10s prompt: %s" % (o, by_sid[k]["prompts"][o]))
    say("")
    say("UNJUDGED SCENARIOS (page only): prompts and clip facts")
    for r in rows:
        if r["type"] and r["sid"] in per:
            continue
        if r["water"] and r["sid"] in water:
            continue
        say(" %s (%s, %s): %s" % (r["sid"], r["model"], r["group"], r["no_judge"] or "no judge"))
        for o in ARMS:
            f = facts[r["sid"]][o]
            say("    %-10s -40 dB after %.2f s, length %.2f s, static %d | %s" % (
                o, f["t40"] or 0, f["len"] or 0, f["static"], r["prompts"][o]))
    with open(os.path.join(a.work, "report%s.txt" % tag(a)), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("saved", os.path.join(a.work, "report%s.json" % tag(a)))
    return 0


def joined(parts):
    J, W = {}, {}
    for work in parts:
        rows = load(scen_path(work))["rows"]
        V = load(os.path.join(work, "vocab.json"))
        ge.merge_vocab(V, work)
        WM = water_merged(work)
        for out, part in ((J, vocab_rows(V, rows)), (W, water_rows(WM, rows) if WM else {})):
            for k, r in part.items():
                if k not in out:
                    out[k] = dict(r, arms={a: {f: list(h) for f, h in r["arms"][a].items()} for a in ARMS})
                    continue
                if out[k]["variants"] != r["variants"]:
                    raise SystemExit("%s: judges differ between %s and the first part" % (k, work))
                for a in ARMS:
                    for f, h in r["arms"][a].items():
                        out[k]["arms"][a][f] += list(h)
    return J, W


def cmd_combine(a):
    CMP = cmp_pairs(ARMS)
    ge.RUN, ge.CMP = ARMS, CMP
    rng = np.random.default_rng(ge.BOOT_SEED)
    J, W = joined(a.parts)
    G = {n: ge.group_stats(m, J, rng) for n, m in groups_of(J).items()}
    if W:
        G["water"] = ge.group_stats(list(W), W, rng)
        AW = dict(J, **{"water:" + k: r for k, r in W.items()})
        G["all_plus_water"] = ge.group_stats(list(AW), AW, rng)
    per = {k: {"cons": {arm: cons(J, k, arm) for arm in ARMS},
               "cmp": {"%s vs %s" % (x, y): ge.paired_key(J[k], x, y, rng) for x, y in CMP}} for k in J}
    wper = {k: {"cons": {arm: cons(W, k, arm) for arm in ARMS},
                "cmp": {"%s vs %s" % (x, y): ge.paired_key(W[k], x, y, rng) for x, y in CMP}} for k in W}
    n = len(next(iter(next(iter(J.values()))["arms"]["phrasebook"].values())))
    dump({"parts": a.parts, "seeds_per_row": n, "groups": G, "per": per, "water": wper}, a.out)
    print("COMBINED over %s: %d seeds per row" % (", ".join(a.parts), n))
    for g, s in G.items():
        print(" %s, %d rows: %s" % (GROUP_TEXT.get(g, g), len(s["keys"]), "  ".join(
            "%s %s" % (arm, fmt_ci(s["arms"][arm]["mean"], s["arms"][arm]["ci"], False)) for arm in ARMS)))
        for name, c in s["cmp"].items():
            print("    %-26s %-26s p %.3f  rows up/down %d/%d  judges %s" % (
                name, fmt_ci(c["delta"], c["ci"]), c["p_boot"], c["up"], c["down"], " ".join(
                    "%s +%d/-%d" % (f[0], x["gain"], x["loss"]) for f, x in c["judges"].items())))
    print("saved", a.out)
    return 0


PAGE_TEXT = {"title": "Prompt Writer Listening", "id": "llm", "intro": """<h1>Prompt writer: blind listening</h1>
<p>Use headphones. Each row is one moment a game sends, rendered on the models that ship: SA3 medium for whatever the
route calls a gun, Woosh DFlow for everything else. The options are three prompt sources for the same request on the
same seeds: the shipped phrasebook, the universal caption template, and the prompt the small CPU language model wrote
from the scenario. Two sources that wrote the same text share one option, so a row has two or three options. Every
option plays %TAKES% takes (variants 40 to 43, which no earlier run rendered). Letters are shuffled per row.</p>
<p>The first two sections are the 40 bench scenarios, which carry the scene record (who or what made the sound, the
object, what it touched, the room); the facts under each title are what the writer read. The last two are the 35 plain
requests of the generic run, where the writer saw only the event and the material. Levels are what the game plays.
Mark the option that sounds most like the moment and the one that sounds least like it, or press "none sound right".
Picks stay in this browser. Reveal names the prompt source behind each option, the model, the prompt and the judge
score where a validated judge exists; Download saves your picks with the key.</p>"""}
PAGE_SECTIONS = (("bench", "demo", "Demo moments with the scene record"), ("bench", "unseen", "Other moments with the scene record"),
                 ("generic", "demo", "Demo requests, plain keys"), ("generic", "unseen", "Unseen requests, plain keys"))


def row_title(r):
    k = r["key"]
    head = "%s | %s" % (k["event_type"], k["material"]) if k["material"] else k["event_type"]
    if r["set"] != "bench":
        return head, ""
    facts = "; ".join(ln for ln in r["text"].split("\n")[1:])
    return head, facts


def cmd_page(a):
    rep = load(os.path.join(a.work, "report%s.json" % tag(a)))
    rows = scenarios(a.work)
    meta = {(x["cid"], x["v"]): x for x in render_rows(a.work)}
    out = a.out
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(os.path.join(out, "a"))
    key, info, n_rows, n_clips = {}, {}, 0, 0
    sec = {(s, g): [] for s, g, _t in PAGE_SECTIONS}
    for r in rows:
        sid = r["sid"]
        opts = {}
        for arm in ARMS:
            opts.setdefault(arm_clip(r, arm), []).append(arm)
        order = sorted(opts, key=lambda c: ge.h(a.seed + sid + c))
        blocks = []
        for i, cid in enumerate(order):
            arms = opts[cid]
            clips, nums = [], []
            for v in range(a.first_seed, a.first_seed + a.takes):
                y = np.load(npy_path(a.work, cid, v))
                x = meta[(cid, v)]
                nm = ge.h(a.seed + sid + cid + str(v))[:12] + ".wav"
                AudioClip(np.ascontiguousarray(y, dtype=np.float32), int(x["sr"])).to_wav(os.path.join(out, "a", nm))
                n_clips += 1
                nums.append(x)
                clips.append("<audio controls preload='none' src='a/%s' aria-label='%s take %d'></audio>" % (
                    nm, ge.LETTERS[i], v - a.first_seed + 1))
            letter = ge.LETTERS[i]
            key["%s|%s" % (sid, letter)] = arms
            judged_ = []
            pk = rep["per"].get(sid)
            if pk:
                judged_.append("identity %.2f [%.2f, %.2f]" % (pk["cons"][arms[0]], pk["cons_ci"][arms[0]][0],
                                                              pk["cons_ci"][arms[0]][1]))
            wk = rep["water"].get(sid)
            if wk:
                judged_.append("water %.2f [%.2f, %.2f]" % (wk["cons"][arms[0]], wk["cons_ci"][arms[0]][0],
                                                           wk["cons_ci"][arms[0]][1]))
            info["%s|%s" % (sid, letter)] = "%s | %s, amp %s | %s | takes (variants %d-%d): -40 dB after %s s, length %s s | prompt: %s" % (
                ", ".join(arms), ge.MODEL_NAME[r["model"]], AMP, "; ".join(judged_) or "no validated judge for this sound",
                a.first_seed, a.first_seed + a.takes - 1, " / ".join("%.2f" % n["t40"] for n in nums),
                " / ".join("%.2f" % n["len"] for n in nums), nums[0]["prompt"])
            blocks.append("<div class='opt' data-s='%s' data-l='%s'><div class='let'>%s</div><div class='takes'>%s</div>"
                          "<div class='btns'><button data-v='best'>best</button><button data-v='worst'>worst</button>"
                          "</div><div class='rev'></div></div>" % (sid, letter, letter, "".join(clips)))
        head, facts = row_title(r)
        sub = "<p class='facts'>%s</p>" % html.escape(facts) if facts else ""
        sec[(r["set"], r["group"])].append(
            "<section class='snd' data-s='%s'><h3>%s</h3>%s%s<p class='row'><button class='none' data-s='%s'>none sound "
            "right</button><span class='st'></span></p></section>" % (sid, html.escape(head), sub, "".join(blocks), sid))
        n_rows += 1
    body = "".join("<h2>%s</h2>%s" % (t, "".join(sec[(s, g)])) for s, g, t in PAGE_SECTIONS if sec[(s, g)])
    payload = base64.b64encode(json.dumps({"key": key, "info": info, "arms": ARM_TEXT}, ensure_ascii=True).encode()).decode()
    page = ge.fill_page(PAGE_TEXT, body, payload, n_rows, "sarge_llm_page_" + a.seed, a.takes)
    page = page.replace("</style>", ".facts{font-size:13px;margin:0 0 6px;overflow-wrap:anywhere}</style>", 1)
    with open(os.path.join(out, "index.html"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(page)
    dump({"seed": a.seed, "key": key, "info": info}, os.path.join(os.path.dirname(os.path.abspath(out)), "page_key.json"))
    print("wrote %s (%d rows, %d clips)" % (os.path.join(out, "index.html"), n_rows, n_clips), flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="the prompt writer against the phrasebook and the caption template on the "
                                             "models that ship: write, render, judge, report, blind page")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("write", "variants", "render-woosh", "render-sa3", "routes", "judge-vocab", "judge-water", "report",
                 "page", "combine"):
        p = sub.add_parser(name)
        p.add_argument("--arms", nargs="+", choices=ALL_ARMS, default=list(BASE_ARMS))
        p.add_argument("--tag", default="")
        p.add_argument("--work", default=WORK)
        p.add_argument("--first-seed", type=int, default=FIRST)
        p.add_argument("--seeds", type=int, default=SEEDS)
        p.add_argument("--takes", type=int, default=TAKES)
        p.add_argument("--water-dir", default=WATER_DIR)
        if name == "write":
            p.add_argument("--model", default=wr.DEFAULT_MODEL)
            p.add_argument("--port", type=int, default=11546)
            p.add_argument("--threads", type=int, default=4)
        if name in ("render-woosh", "render-sa3"):
            p.add_argument("--only", nargs="+", default=None)
        if name == "judge-vocab":
            p.add_argument("--refs", default=os.path.join("data", "refs_heldout"))
        if name == "page":
            p.add_argument("--out", default=PAGE)
            p.add_argument("--seed", default="llm1")
        if name == "combine":
            p.add_argument("--parts", nargs="+", required=True, help="work folders whose judged seeds are pooled per row")
            p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    global ARMS
    ARMS = tuple(a.arms)
    a.water_dir = os.path.abspath(os.path.join(ROOT, a.water_dir))
    return {"write": cmd_write, "variants": cmd_variants, "render-woosh": cmd_render_woosh, "render-sa3": cmd_render_sa3, "routes": cmd_routes,
            "judge-vocab": cmd_judge_vocab, "judge-water": cmd_judge_water, "report": cmd_report,
            "page": cmd_page, "combine": cmd_combine}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())

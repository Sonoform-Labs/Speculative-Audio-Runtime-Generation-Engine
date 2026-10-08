import os
import sys

BENCH = os.path.join(os.getcwd(), "scripts", "bench")
if not os.path.isfile(os.path.join(BENCH, "_bootstrap.py")):
    BENCH = r"D:\Python\SAGE\scripts\bench"
sys.path.insert(0, BENCH)

import _bootstrap

import argparse
import gc
import json
import re
import time
from collections import Counter, defaultdict

import numpy as np
from scipy.io import wavfile

from sarge.generators.stable_audio import _extract, build_prompt
from sarge.generators.stable_audio3 import StableAudio3Generator
from sarge.loudness import normalize_loudness, target_for
from sarge.types import AudioClip, CacheKey
from validate_panel import _resample, judge_audiobox
from recalibrate import attack_ms, auc, crest_db, softmax
from ab_clap import clap_scores, mcnemar
from water_judge import HISS_PER, HISS_SPECTRA, TAG_FITS, _release, _torch, frac, hiss, pct, resolve, run_ast, \
    run_panns, slug

ROOT = os.path.dirname(os.path.dirname(BENCH))
SR = 44100
INTENSITY = "normal"
WINS = ["0.6", "1.5"]
PRODUCTION = {"sampler_type": "euler", "steps": 3}
SETTINGS = {
    "s3_g1.0_k0.6": {"steps": 3, "detail_steps": 3, "gen_duration_s": 1.0, "duration_s": 0.6},
    "s8_g1.0_k0.6": {"steps": 8, "detail_steps": 8, "gen_duration_s": 1.0, "duration_s": 0.6},
    "s3_g2.0_k1.5": {"steps": 3, "detail_steps": 3, "gen_duration_s": 2.0, "duration_s": 1.5},
    "s8_g2.0_k1.5": {"steps": 8, "detail_steps": 8, "gen_duration_s": 2.0, "duration_s": 1.5},
}
CURRENT = "s3_g1.0_k0.6"
MANIFEST = os.path.join("data", "refs", "manifest.json")
ROUTE_NAME = re.compile(r"[A-Za-z0-9_.+-]+")
SURFACES = ["wood", "stone", "grass", "metal", "gravel"]
KEYS = [
    ("shoot", "rifle", "gunshot_rifle"), ("shoot", "pistol", "gunshot_pistol"), ("reload", "rifle", "gun_mech"),
    ("equip", "rifle", None), ("empty_click", "pistol", "gun_mech"), ("explosion", "grenade", "explosion"),
    ("swing", "sword", "sword"), ("open", "door", "door_open"), ("slam", "door", "door_slam"),
    ("break", "window", "glass_break"), ("click", "button", "ui_click"), ("cast", "fire_spell", None),
    ("punch", "body", "punch"), ("horn", "truck", "horn"), ("bark", "dog", "dog_bark"),
] + [("footstep", s, "footstep") for s in SURFACES]
GUN = {"gun": ["Gunshot, gunfire", "Machine gun", "Fusillade", "Cap gun"], "gunshot": ["Gunshot, gunfire"]}
FOOT = {"foot": ["Walk, footsteps"], "loco": ["Walk, footsteps", "Run", "Shuffle"]}
TAG_SETS = {
    "gunshot_pistol": GUN, "gunshot_rifle": GUN, "gunshot_shotgun": GUN, "gunshot_far": GUN,
    "gun_mech": {"mech": ["Mechanisms", "Ratchet, pawl", "Clicking", "Clickety-clack", "Clatter"],
                 "clicking": ["Clicking"]},
    "shell_drop": {"clink": ["Chink, clink", "Jingle, tinkle", "Coin (dropping)", "Clatter"]},
    "explosion": {"explosion": ["Explosion", "Boom", "Burst, pop", "Fireworks", "Firecracker", "Eruption",
                                "Artillery fire"], "boom": ["Explosion", "Boom"]},
    "sword": {"blade": ["Clang", "Ding", "Ping", "Zing", "Chink, clink", "Scrape", "Whoosh, swoosh, swish"],
              "swing": ["Whoosh, swoosh, swish"]},
    "door_open": {"door": ["Door", "Sliding door", "Squeak", "Creak", "Cupboard open or close", "Drawer open or close"],
                  "door_only": ["Door"]},
    "door_slam": {"slam": ["Slam", "Door"], "slam_only": ["Slam"]},
    "glass_break": {"glass": ["Glass", "Shatter", "Breaking", "Chink, clink", "Smash, crash"],
                    "shatter": ["Shatter", "Glass"]},
    "ui_click": {"click": ["Clicking", "Computer keyboard", "Typing", "Tick", "Clickety-clack"],
                 "clicking": ["Clicking"]},
    "punch": {"hit": ["Slap, smack", "Whack, thwack", "Thump, thud", "Thunk"], "slap": ["Slap, smack", "Whack, thwack"]},
    "horn": {"horn": ["Vehicle horn, car horn, honking", "Air horn, truck horn", "Toot", "Train horn", "Foghorn"],
             "vehicle": ["Vehicle horn, car horn, honking", "Air horn, truck horn"]},
    "dog_bark": {"dog": ["Dog", "Bark", "Yip", "Howl", "Bow-wow", "Growling", "Whimper (dog)", "Canidae, dogs, wolves"],
                 "bark": ["Bark", "Bow-wow"]},
    "splash": {"splash": ["Splash, splatter", "Slosh", "Plop"]},
    "footstep": FOOT, "footstep_esc": FOOT,
}
PHRASES = {
    "gunshot_pistol": ["a pistol gunshot", "a handgun firing", "a gunshot"],
    "gunshot_rifle": ["a rifle gunshot", "a rifle firing", "a gunshot"],
    "gunshot_shotgun": ["a shotgun blast", "a shotgun firing", "a gunshot"],
    "gunshot_far": ["a distant gunshot", "a gunshot"],
    "gun_mech": ["a gun being reloaded", "an empty gun clicking", "a gun mechanism", "a gun slide being racked"],
    "shell_drop": ["bullet casings dropping", "small metal objects dropping"],
    "explosion": ["a grenade explosion", "an explosion", "a loud blast"],
    "sword": ["a sword swinging through the air", "swords clashing", "a sword being drawn", "a metal blade"],
    "door_open": ["a door opening", "a creaking door", "a door"],
    "door_slam": ["a door slamming shut", "a door slam", "a door"],
    "glass_break": ["a window breaking", "glass shattering", "breaking glass"],
    "ui_click": ["a user interface click", "a mouse click", "a button click"],
    "punch": ["a punch hitting a body", "a punch", "a body blow"],
    "horn": ["a truck horn", "a car horn", "a vehicle horn honking"],
    "dog_bark": ["a dog barking", "a dog bark", "a dog"],
    "splash": ["water splashing", "a splash"],
    "footstep": ["a footstep", "footsteps", "a single footstep"],
    "footstep_esc": ["a footstep", "footsteps"],
}
CANON = {
    "gunshot_pistol": ["a pistol gunshot"], "gunshot_rifle": ["a rifle gunshot"], "gunshot_shotgun": ["a shotgun blast"],
    "gunshot_far": ["a distant gunshot"], "gun_mech": ["a gun being reloaded", "an empty gun clicking"],
    "shell_drop": ["bullet casings dropping"], "explosion": ["a grenade explosion"],
    "sword": ["a sword swinging through the air", "swords clashing", "a sword being drawn"],
    "door_open": ["a door opening"],
    "door_slam": ["a door slamming shut"], "glass_break": ["a window breaking"], "ui_click": ["a user interface click"],
    "punch": ["a punch hitting a body"], "horn": ["a truck horn"], "dog_bark": ["a dog barking"],
    "splash": ["water splashing"], "footstep": ["a footstep"], "footstep_esc": ["a footstep"],
}
EXTRA = ["a weapon being drawn", "a magic fire spell"]
SAME_EVENT = {"sword": ["a weapon being drawn"]}
GENERIC = ["white noise static", "wind", "fire crackling", "water dripping", "a knock", "music", "speech"]
MAT_TEXT = "the sound of a footstep on %s"
RULES = ("top", "thr")
JUDGE_FAMS = ("ast", "panns", "clap")
AB_AXES = ("CE", "CU", "PC", "PQ")
SHAPE = ("tail_ms", "decay100_db", "low500", "crest_db", "attack_ms", "floor_db")
NAN = float("nan")


def dedupe(xs):
    out = []
    for x in xs:
        if x not in out:
            out.append(x)
    return out


CLAP_TEXTS = dedupe([p for t in PHRASES for p in PHRASES[t]] + [p for t in CANON for p in CANON[t]] + EXTRA + GENERIC +
                    [MAT_TEXT % s for s in SURFACES])


def rel(p):
    try:
        return os.path.relpath(p, ROOT).replace(os.sep, "/")
    except ValueError:
        return os.path.abspath(p).replace(os.sep, "/")


def wav(p):
    c = AudioClip.from_wav(p)
    x = c.samples if c.sample_rate == SR else _resample(c.samples, c.sample_rate, SR)
    return np.ascontiguousarray(x, dtype=np.float32)


def level(x):
    return normalize_loudness(x, target_rms=target_for(INTENSITY))


def key_name(et, mat):
    return "%s|%s" % (et, mat)


def gen_name(s, et, mat):
    return "gen:%s|%s|%s" % (s, et, mat)


def cat(xs):
    xs = [np.asarray(x) for x in xs]
    return np.concatenate(xs) if xs else np.array([])


def ci(v):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v.size else [NAN, NAN]


def spaced(items, k):
    if k is None or len(items) <= k:
        return list(items)
    idx = np.unique(np.round(np.linspace(0, len(items) - 1, k)).astype(int))
    return [items[i] for i in idx]


def free_gpu(tag=""):
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("gpu after %s: %.0f MB allocated, %.0f MB reserved" % (
            tag or "free", torch.cuda.memory_allocated() / 2 ** 20, torch.cuda.memory_reserved() / 2 ** 20), flush=True)


class Boot:
    def __init__(self, reps, seed):
        self.reps = int(reps)
        self.rng = np.random.default_rng(seed)
        self.cache = {}

    def draws(self, tag, clusters):
        if tag not in self.cache:
            keys, inv = np.unique(np.asarray(clusters, dtype=str), return_inverse=True)
            k = len(keys)
            w = self.rng.multinomial(k, np.full(k, 1.0 / k), size=self.reps).astype(np.float32)
            n = np.bincount(inv, minlength=k).astype(np.float32)
            self.cache[tag] = (inv, w, w @ n, len(clusters))
        inv, w, wn, size = self.cache[tag]
        if size != len(clusters):
            raise SystemExit("bootstrap tag %s reused with %d clips, built with %d" % (tag, len(clusters), size))
        return inv, w, wn

    def rate(self, tag, clusters, h):
        inv, w, wn = self.draws(tag, clusters)
        s = np.bincount(inv, weights=np.asarray(h, dtype=np.float64), minlength=w.shape[1]).astype(np.float32)
        return (w @ s) / np.maximum(wn, 1e-12)

    def seeds(self, n):
        k = ("seeds", n)
        if k not in self.cache:
            self.cache[k] = self.rng.integers(0, n, size=(self.reps, n))
        return self.cache[k]


def load_refs(args, cfgs):
    man = json.load(open(os.path.join(ROOT, args.manifest), encoding="utf-8"))
    missing = [t for t in man["types"] if t not in TAG_SETS or t not in PHRASES or t not in CANON]
    if missing:
        raise SystemExit("types in the manifest without judge definitions: %s" % missing)
    mm = man["meta"]
    if int(mm["sr"]) != SR or mm["intensity"] != INTENSITY:
        raise SystemExit("manifest built at %s Hz, intensity %s" % (mm["sr"], mm["intensity"]))
    for s, cfg in cfgs.items():
        ln = mm["lengths"].get("%.1f" % cfg["duration_s"])
        if not ln or abs(ln["keep_s"] - cfg["duration_s"]) > 1e-9 or abs(ln["gen_s"] - cfg["gen_duration_s"]) > 1e-9:
            raise SystemExit("setting %s keeps %.1f s of %.1f s, manifest references are %s" % (
                s, cfg["duration_s"], cfg["gen_duration_s"], ln))
    by = defaultdict(list)
    for c in man["clips"]:
        by[c["set"]].append(c)
    sets, meta = {}, {}
    for s, spec in man["sets"].items():
        cs = by.get(s, [])
        if not cs:
            continue
        for w in WINS:
            name = "ref:%s@%s" % (s, w)
            sets[name] = [wav(os.path.join(ROOT, *c["files"][w].split("/"))) for c in cs]
            meta[name] = {"kind": "ref", "set": s, "type": spec["type"], "family": spec["family"], "win": w,
                          "dataset": spec["dataset"], "license": spec["license"], "ids": [c["id"] for c in cs],
                          "clusters": [c["cluster"] for c in cs], "strata": [c["stratum"] for c in cs],
                          "surface": [c.get("surface") for c in cs]}
    return man, sets, meta


def build_hiss(sets, meta):
    spec = [(a, hp) for a, hp in HISS_SPECTRA for _ in range(HISS_PER)]
    raw = [hiss(1000 + j, a, hp) for j, (a, hp) in enumerate(spec)]
    for w in WINS:
        name = "hiss@%s" % w
        sets[name] = [level(_extract(x, SR, float(w))) for x in raw]
        meta[name] = {"kind": "hiss", "type": None, "family": "hiss", "win": w,
                      "ids": ["hiss_%02d" % j for j in range(len(raw))],
                      "clusters": ["hiss:%02d" % j for j in range(len(raw))], "spectra": spec}


def make_generator():
    return StableAudio3Generator(**PRODUCTION)


def gen_info(g):
    info = {a: getattr(g, a, None) for a in ("name", "model_name", "model_half", "cfg_scale", "sampler_type", "steps",
                                             "detail_steps", "gen_duration_s", "duration_s", "duration_padding_s",
                                             "fast", "hoist_prompts", "static_guard", "max_static_retries",
                                             "sample_rate")}
    info["detail_keys"] = sorted(getattr(g, "detail_keys", []) or [])
    info["accelerated"] = getattr(g, "_fast", None) is not None
    info["hoisted"] = getattr(g, "_hoist", None) is not None
    return info


def gen_paths(args, s, et, mat):
    return [os.path.join(ROOT, args.gen_dir, s, "%s-%s_%02d.wav" % (et, mat, v)) for v in range(args.seeds)]


def norm(p):
    return os.path.normcase(os.path.abspath(p))


def route_leaks(info, man):
    used_c = set(info.get("clusters", []))
    srcs = [p if os.path.isabs(p) else os.path.join(ROOT, p) for p in info.get("sources", [])]
    used_s = {norm(p) for p in srcs}
    held = man["meta"].get("heldout") or {}
    hc = {c for v in held.get("clusters", {}).values() for c in v}
    files = {norm(c["src"]) for c in man["clips"]} | {norm(os.path.join(ROOT, *f.split("/"))) for c in man["clips"]
                                                      for f in c.get("files", {}).values()}
    ids = {"%s.wav" % c["id"].lower() for c in man["clips"]}
    esc = norm(os.path.join(ROOT, "data", "esc50"))
    return {"clusters_in_refs": sorted(used_c & {c["cluster"] for c in man["clips"]}),
            "sources_in_refs": sorted((used_s & files) | {p for p in used_s if os.path.basename(p) in ids}),
            "clusters_held_out": sorted(used_c & hc),
            "esc50_used": sorted({c for c in used_c if c.startswith("esc:")} |
                                 {p for p in used_s if p == esc or p.startswith(esc + os.sep)})}


def load_routes(args, man):
    cfgs, sets, meta, runs = {}, {}, {}, {}
    lens = {int(float(w) * SR): w for w in man["meta"]["lengths"]}
    held = bool(man["meta"].get("heldout"))
    for spec in args.route:
        name, _, d = spec.partition("=")
        root = d if os.path.isabs(d) else os.path.join(ROOT, d)
        if not os.path.isdir(root):
            raise SystemExit("route %s: no folder %s" % (name, root))
        jp = os.path.join(root, "route.json")
        info = json.load(open(jp, encoding="utf-8")) if os.path.isfile(jp) else None
        if held and (info is None or "clusters" not in info):
            raise SystemExit("route %s: %s needs a route.json whose clusters list every source it trained on or "
                             "anchored from (empty for a route that uses no real audio) to be scored on held-out "
                             "references" % (name, rel(root)))
        leak = route_leaks(info or {}, man)
        if any(leak.values()):
            if held:
                raise SystemExit("route %s used held-out sources: %s" % (name, json.dumps(leak)))
            print("route %s shares sources with %s: %d clusters, %d source files, %d ESC-50 sources" % (
                name, args.manifest, len(leak["clusters_in_refs"]), len(leak["sources_in_refs"]),
                len(leak["esc50_used"])), flush=True)
        win = None
        for et, mat, t in KEYS:
            ps = [os.path.join(root, "%s-%s_%02d.wav" % (et, mat, v)) for v in range(args.seeds)]
            have = sum(os.path.isfile(p) for p in ps)
            if not have:
                continue
            if have < len(ps):
                raise SystemExit("route %s has %d of %d seeds for %s" % (name, have, len(ps), key_name(et, mat)))
            clips = [wav(p) for p in ps]
            n = sorted({len(x) for x in clips})
            if len(n) != 1 or n[0] not in lens or lens[n[0]] not in WINS:
                raise SystemExit("route %s %s clips have lengths %s, expected one of %s" % (
                    name, key_name(et, mat), n, sorted(lens)))
            if win is not None and lens[n[0]] != win:
                raise SystemExit("route %s mixes %s s and %s s clips" % (name, win, lens[n[0]]))
            win = lens[n[0]]
            g = gen_name(name, et, mat)
            sets[g] = clips
            meta[g] = {"kind": "gen", "setting": name, "key": key_name(et, mat), "type": t, "family": None, "win": win,
                       "seeds": list(range(args.seeds)),
                       "ids": ["%s#%02d" % (key_name(et, mat), v) for v in range(args.seeds)],
                       "clusters": ["seed:%02d" % v for v in range(args.seeds)], "paths": [rel(p) for p in ps],
                       "prompt": (info or {}).get("prompts", {}).get(key_name(et, mat)),
                       "steps": (info or {}).get("steps"), "reused": True, "route": name}
            runs[g] = {"ms_median": None, "ms_max": None, "static": {}}
        if win is None:
            raise SystemExit("route %s: no <event>-<material>_NN.wav files for any key in %s" % (name, rel(root)))
        cfgs[name] = {"route": rel(root), "duration_s": float(win), "gen_duration_s": man["meta"]["lengths"][win]["gen_s"],
                      "steps": (info or {}).get("steps"), "keys": sum(meta[g]["setting"] == name for g in meta),
                      "route_json": info is not None, "leak": leak, "info": info}
        print("route %-13s %d keys x %d seeds at %s s from %s%s" % (
            name, cfgs[name]["keys"], args.seeds, win, rel(root), "" if info is not None else "  (no route.json)"),
            flush=True)
    return cfgs, sets, meta, runs


def generate(args, cfgs):
    if not cfgs:
        print("generator not loaded: no base settings", flush=True)
        return {}, {}, {}, {"loaded": False, "reason": "no base settings"}
    need = not args.reuse or any(not os.path.isfile(p) for s in cfgs for et, mat, _t in KEYS
                                 for p in gen_paths(args, s, et, mat))
    g = make_generator() if need else None
    if g is not None:
        g.ensure_ready()
        if int(g.sample_rate) != SR:
            raise SystemExit("generator sample rate %s, expected %d" % (g.sample_rate, SR))
        info = gen_info(g)
    else:
        info = {"loaded": False, "reason": "--reuse with every file on disk"}
    print("generator %s" % json.dumps(info, default=str), flush=True)
    keys = [CacheKey(et, mat, INTENSITY) for et, mat, _t in KEYS]
    sets, meta, runs = {}, {}, {}
    for s, cfg in cfgs.items():
        if g is not None:
            for a, v in cfg.items():
                setattr(g, a, v)
            for k in keys:
                if g._steps_for(k) != cfg["steps"]:
                    raise SystemExit("steps for %s = %d under %s, expected %d" % (k, g._steps_for(k), s, cfg["steps"]))
                if g._prompt(k) != build_prompt(k.event_type, k.material, k.intensity):
                    raise SystemExit("prompt for %s differs from build_prompt" % k)
        os.makedirs(os.path.join(ROOT, args.gen_dir, s), exist_ok=True)
        warm = False
        t_set = time.perf_counter()
        for (et, mat, t), k in zip(KEYS, keys):
            ps = gen_paths(args, s, et, mat)
            reuse = bool(args.reuse and all(os.path.isfile(p) for p in ps))
            if not reuse and not warm:
                g.generate(keys[0], variant=0)
                warm = True
            st0 = dict(getattr(g, "static_stats", {}) or {})
            ms = []
            if not reuse:
                for v, p in enumerate(ps):
                    t0 = time.perf_counter()
                    c = g.generate(k, variant=v)
                    ms.append((time.perf_counter() - t0) * 1000.0)
                    c.to_wav(p)
            st1 = dict(getattr(g, "static_stats", {}) or {})
            clips = [wav(p) for p in ps]
            n = int(cfg["duration_s"] * SR)
            bad = sorted({len(x) for x in clips if len(x) != n})
            if bad:
                raise SystemExit("%s %s clips have lengths %s, expected %d" % (s, k, bad, n))
            name = gen_name(s, et, mat)
            sets[name] = clips
            meta[name] = {"kind": "gen", "setting": s, "key": key_name(et, mat), "type": t, "family": None,
                          "win": "%.1f" % cfg["duration_s"], "seeds": list(range(args.seeds)),
                          "ids": ["%s#%02d" % (key_name(et, mat), v) for v in range(args.seeds)],
                          "clusters": ["seed:%02d" % v for v in range(args.seeds)],
                          "paths": [rel(p) for p in ps],
                          "prompt": g._prompt(k) if g is not None else build_prompt(et, mat, INTENSITY),
                          "steps": int(g._steps_for(k)) if g is not None else cfg["steps"], "reused": reuse}
            runs[name] = {"ms_median": float(np.median(ms)) if ms else None, "ms_max": float(max(ms)) if ms else None,
                          "static": {a: int(st1.get(a, 0)) - int(st0.get(a, 0)) for a in st1}}
        fs = g.fast_stats() if g is not None and hasattr(g, "fast_stats") else {}
        info.setdefault("fast_stats", {})[s] = fs
        print("generated %-14s %d keys x %d seeds  %.1f s  (%s)  %s" % (
            s, len(KEYS), args.seeds, time.perf_counter() - t_set,
            "reused files" if all(meta[gen_name(s, et, mat)]["reused"] for et, mat, _t in KEYS) else "fresh",
            "  ".join("%s %s" % (k, fs[k]) for k in ("cond_misses", "graphs", "replays", "eager", "capture_failures",
                                                     "installed", "slots", "hoisted", "plain", "slot_failures",
                                                     "hoist_capture_failures", "static_retries") if k in fs)),
            flush=True)
    del g
    free_gpu("generator")
    return sets, meta, runs, info


def run_clap(sets, args):
    torch, dev = _torch(args.threads)
    t0 = time.perf_counter()
    s48 = {k: [_resample(x, SR, 48000) for x in v] for k, v in sets.items()}
    res = clap_scores(s48, {"all": list(CLAP_TEXTS)}, dev, 4 * args.bs)
    print("clap  native %7.1f s for %d clips x %d texts" % (time.perf_counter() - t0, sum(len(v) for v in sets.values()),
                                                           len(CLAP_TEXTS)), flush=True)
    del s48
    _release(torch)
    return {"labels": list(CLAP_TEXTS), "scores": {"native": {k: np.asarray(v["all"], np.float32) for k, v in res.items()}}}


def run_audiobox(sets, args):
    names = [n for n in sets if not n.startswith("hiss@")]
    flat = [x for n in names for x in sets[n]]
    owner = [n for n in names for _ in sets[n]]
    t0 = time.perf_counter()
    r = judge_audiobox(flat, SR, 2 * args.bs)
    rows = defaultdict(list)
    for row, n in zip(r["rows"], owner):
        rows[n].append([float(row[a]) for a in AB_AXES])
    print("audiobox     %7.1f s for %d clips" % (time.perf_counter() - t0, len(flat)), flush=True)
    return {"labels": list(AB_AXES), "scores": {n: np.asarray(v, dtype=np.float32) for n, v in rows.items()}}


JUDGES = {"ast": run_ast, "panns": run_panns, "clap": run_clap, "audiobox": run_audiobox}


def tagger_variants(fam, res, types):
    labels = res["labels"]
    ids, ls = {}, {}
    for t in types:
        ids[t] = resolve(labels, TAG_SETS[t])
        ls[t] = {s: [labels[i] for i in ix] for s, ix in ids[t].items()}
    out = {}
    for f, by in res["scores"].items():
        tops = {n: np.argsort(-P, axis=1)[:, :5] for n, P in by.items()}
        top1 = {n: [labels[j] for j in tops[n][:, 0]] for n in by}
        for t, sets_ in ids.items():
            for s, ix in sets_.items():
                ixa = np.asarray(ix)
                d = {}
                for n, P in by.items():
                    sub = P[:, ixa]
                    d[n] = {"hit": np.isin(tops[n], ixa).any(1), "score": sub.sum(1).astype(float),
                            "lead": [labels[ix[j]] for j in sub.argmax(1)], "top1": top1[n]}
                out["%s|%s|%s|%s" % (fam, f, t, s)] = d
    return out, ls


def clap_variants(res, types, family):
    texts = list(res["labels"])
    col = {x: i for i, x in enumerate(texts)}
    ident = dedupe([p for t in types for p in CANON[t]] + EXTRA + GENERIC)
    icol = [col[p] for p in ident]
    out, ls = {}, {"identity_texts": ident}
    for f, by in res["scores"].items():
        idp = {n: softmax(L[:, icol]) for n, L in by.items()}
        idtop = {n: [ident[j] for j in P.argmax(1)] for n, P in idp.items()}
        for t in types:
            dis = dedupe(GENERIC + EXTRA + [p for u in types if family[u] != family[t] for p in CANON[u]])
            dis = [d for d in dis if d not in PHRASES[t] and d not in SAME_EVENT.get(t, [])]
            groups = {"p_" + slug(p): [p] for p in PHRASES[t]}
            groups["any"] = list(PHRASES[t])
            ls[t] = {"phrases": list(PHRASES[t]), "distractors": dis, "canon": list(CANON[t])}
            for gname, ps in groups.items():
                cols = [col[p] for p in ps] + [col[x] for x in dis]
                d = {}
                for n, L in by.items():
                    P = softmax(L[:, cols])
                    am = P.argmax(1)
                    d[n] = {"hit": am < len(ps), "score": P[:, :len(ps)].sum(1).astype(float),
                            "lead": [ps[int(j)] for j in P[:, :len(ps)].argmax(1)],
                            "top1": [texts[cols[int(j)]] for j in am]}
                out["clap|%s|%s|%s" % (f, t, gname)] = d
            own = [ident.index(p) for p in CANON[t]]
            famc = [ident.index(p) for p in dedupe([p for u in types if family[u] == family[t] for p in CANON[u]] +
                                                   SAME_EVENT.get(t, []))]
            for gname, ix in (("id", own), ("idfam", famc)):
                d = {}
                for n, P in idp.items():
                    d[n] = {"hit": np.isin(P.argmax(1), ix), "score": P[:, ix].sum(1).astype(float),
                            "lead": [ident[ix[int(j)]] for j in P[:, ix].argmax(1)], "top1": idtop[n]}
                out["clap|%s|%s|%s" % (f, t, gname)] = d
    return out, ls


def material_preds(res):
    texts = list(res["labels"])
    cols = [texts.index(MAT_TEXT % s) for s in SURFACES]
    return {n: np.asarray(L[:, cols]).argmax(1) for n, L in res["scores"]["native"].items()}


def neg_pools(meta, k):
    pools = {}
    for w in WINS:
        fams = defaultdict(list)
        for n, m in meta.items():
            if m["kind"] == "ref" and m["win"] == w:
                for i, (c, iid) in enumerate(zip(m["clusters"], m["ids"])):
                    fams[m["family"]].append((m["set"], c, iid, n, i))
        pools[w] = {}
        for f, items in fams.items():
            items.sort(key=lambda r: (r[0], r[1], r[2]))
            pick = spaced(items, k)
            g = defaultdict(list)
            for r in pick:
                g[r[3]].append(r[4])
            pools[w][f] = {"by_set": {n: np.asarray(ix, dtype=int) for n, ix in g.items()},
                           "clusters": [meta[n]["clusters"][i] for n, ix in g.items() for i in ix],
                           "n": len(pick), "sets": sorted({r[0] for r in pick})}
    return pools


def hits(d, rule, tau):
    return np.asarray(d["hit"], dtype=bool) if rule == "top" else np.asarray(d["score"], dtype=float) > tau


def pool_field(d, pool, field):
    return cat([np.asarray(d[n][field])[ix] for n, ix in pool["by_set"].items()])


def pos_sets(meta, t, w):
    return [n for n, m in meta.items() if m["kind"] == "ref" and m["win"] == w and m["type"] == t]


def evaluate(var, meta, pools, family, args, boot):
    ctl, taus = {}, {}
    for base, d in var.items():
        _fam, _fit, t, _lab = base.split("|")
        for w in WINS:
            pos = pos_sets(meta, t, w)
            hz = "hiss@%s" % w
            if not pos or hz not in d or any(n not in d for n in pos):
                continue
            negf = {f: p for f, p in pools[w].items() if f != family[t]}
            rs = cat([d[n]["score"] for n in pos])
            rc = [c for n in pos for c in meta[n]["clusters"]]
            rst = [s for n in pos for s in meta[n]["strata"]]
            rset = [meta[n]["set"] for n in pos for _ in meta[n]["ids"]]
            ns = {f: pool_field(d, p, "score") for f, p in negf.items()}
            pool_s = cat(list(ns.values()))
            pool_c = [c for p in negf.values() for c in p["clusters"]]
            tau = float(np.quantile(pool_s, 1.0 - args.fpr))
            taus[(base, w)] = tau
            hs = np.asarray(d[hz]["score"], dtype=float)
            a = {"pooled": auc(rs, pool_s), "hiss": auc(rs, hs)}
            enough = bool(len(rc) >= args.min_ref_clips and len(set(rc)) >= args.min_ref_sources)
            top1 = Counter(x for n in pos for x in d[n]["top1"]).most_common(3)
            lead = Counter(x for n in pos for x in d[n]["lead"]).most_common(3)
            for rule in RULES:
                rh = cat([hits(d[n], rule, tau) for n in pos]).astype(bool)
                nh = {f: (pool_field(d, p, "hit").astype(bool) if rule == "top" else ns[f] > tau)
                      for f, p in negf.items()}
                ph = cat(list(nh.values())).astype(bool)
                hh = hits(d[hz], rule, tau)
                rr, nr, hr = frac(rh), frac(ph), frac(hh)
                byf = {f: frac(v) for f, v in nh.items()}
                rb = boot.rate("real|%s|%s" % (t, w), rc, rh)
                nb = boot.rate("neg|%s|%s" % (family[t], w), pool_c, ph)
                rci, gci = ci(rb), ci(rb - nb)
                valid = bool(rr >= args.min_real and rr - nr >= args.min_gap)
                hiss_ok = bool(rr - hr >= args.min_gap)
                worst = max(byf, key=byf.get) if byf else None
                ctl.setdefault("%s|%s" % (base, rule), {})[w] = {
                    "real": rr, "real_ci": rci, "neg": nr, "neg_ci": ci(nb), "gap_ci": gci, "hiss": hr,
                    "neg_by_family": byf, "worst_family": worst, "auc": a, "tau": tau,
                    "n_real": int(len(rh)), "k_real": int(len(set(rc))), "n_neg": int(len(ph)),
                    "k_neg": int(len(set(pool_c))),
                    "real_by_set": {s: frac(rh[np.array([x == s for x in rset])]) for s in sorted(set(rset))},
                    "real_by_stratum": {s: frac(rh[np.array([x == s for x in rst])]) for s in sorted(set(rst))},
                    "real_top1": top1, "real_lead": lead,
                    "hiss_top1": Counter(d[hz]["top1"]).most_common(3),
                    "valid": valid, "hiss_ok": hiss_ok, "usable": bool(valid and hiss_ok and enough),
                    "strict": bool(rr >= args.min_real and byf and rr - max(byf.values()) >= args.min_gap),
                    "firm": bool(valid and hiss_ok and enough and rci[0] >= args.min_real and gci[0] >= args.min_gap)}
    return ctl, taus


def choose(ctl, types):
    chosen = {}
    for t in types:
        for w in WINS:
            pick = {}
            for v, by in ctl.items():
                fam, _fit, tt, _lab, _rule = v.split("|")
                if tt != t or w not in by or not by[w]["usable"]:
                    continue
                c = by[w]
                k = (c["firm"], c["strict"], c["auc"]["pooled"], c["real"] - c["neg"])
                if fam not in pick or k > pick[fam][0]:
                    pick[fam] = (k, v)
            chosen["%s@%s" % (t, w)] = {f: v for f, (_k, v) in sorted(pick.items())}
    return chosen


def real_hits(var, meta, base, rule, tau, t, w):
    pos = pos_sets(meta, t, w)
    return cat([hits(var[base][n], rule, tau) for n in pos]).astype(bool), [c for n in pos for c in meta[n]["clusters"]]


def score_gen(var, ctl, taus, meta, chosen, pools, family, args, boot):
    out, cons = {}, {}
    for g, m in meta.items():
        if m["kind"] != "gen" or not m["type"]:
            continue
        t, w = m["type"], m["win"]
        rec = {}
        for v, by in ctl.items():
            fam, _fit, tt, _lab, rule = v.split("|")
            if tt != t or w not in by or not by[w]["usable"]:
                continue
            base = v.rsplit("|", 1)[0]
            tau = taus[(base, w)]
            d = var[base][g]
            h = hits(d, rule, tau)
            c = by[w]
            negs = {f: p for f, p in pools[w].items() if f != family[t]}
            pool_s = cat([pool_field(var[base], p, "score") for p in negs.values()])
            rec[v] = {"rate": frac(h), "real": c["real"], "ratio": frac(h) / c["real"] if c["real"] > 0 else NAN,
                      "auc_vs_neg": auc(np.asarray(d["score"], float), pool_s),
                      "lead": Counter(x for x, y in zip(d["lead"], h) if y).most_common(3),
                      "top1": Counter(d["top1"]).most_common(3), "hits": [int(x) for x in h]}
        out[g] = rec
        ch = chosen.get("%s@%s" % (t, w), {})
        if not ch:
            cons[g] = {"cons": None, "judges": {}, "reason": "no valid judge for %s at %s s" % (t, w)}
            continue
        idx = boot.seeds(len(m["seeds"]))
        cb, ratios, per = [], [], {}
        for f, v in ch.items():
            base, rule = v.rsplit("|", 1)
            rh, rc = real_hits(var, meta, base, rule, taus[(base, w)], t, w)
            rb = boot.rate("real|%s|%s" % (t, w), rc, rh)
            h = np.asarray(rec[v]["hits"], dtype=float)
            gb = h[idx].mean(1)
            r = np.where(rb > 0, gb / np.maximum(rb, 1e-12), np.nan)
            cb.append(np.minimum(r, 1.0))
            ratios.append(rec[v]["ratio"])
            per[f] = {"variant": v, "rate": rec[v]["rate"], "real": rec[v]["real"], "ratio": rec[v]["ratio"],
                      "ratio_ci": ci(r)}
        cmat = np.vstack(cb)
        cpt = float(min(min(x, 1.0) for x in ratios))
        cons[g] = {"cons": cpt, "cons_ci": ci(np.min(cmat, axis=0)), "mean_ratio": frac(ratios), "judges": per}
    return out, cons


def material(mpred, meta, man, args, boot):
    built = list(man.get("surfaces", {}))
    ctl, gen = {}, {}
    for w in WINS:
        n = "ref:fs_step@%s" % w
        if n not in mpred:
            continue
        surf = np.array([s if s in built else "" for s in meta[n]["surface"]])
        clus = np.array(meta[n]["clusters"])
        pred = np.array([SURFACES[int(i)] for i in mpred[n]])
        ctl[w] = {}
        for s in built:
            on, off = surf == s, (surf != "") & (surf != s)
            h = pred[on] == s
            rb = boot.rate("mat|%s|%s" % (s, w), list(clus[on]), h)
            rr, nr = frac(h), frac(pred[off] == s)
            ctl[w][s] = {"real": rr, "real_ci": ci(rb), "neg": nr, "n": int(on.sum()), "k": int(len(set(clus[on]))),
                         "confusions": Counter(pred[on]).most_common(), "valid": bool(rr >= args.min_real and
                                                                                    rr - nr >= args.min_gap)}
        allb = surf != ""
        ctl[w]["_all"] = {"accuracy": frac(pred[allb] == surf[allb]), "n": int(allb.sum()), "chance": 1.0 / len(SURFACES),
                          "macro": frac([ctl[w][s]["real"] for s in built])}
    for g, m in meta.items():
        if m["kind"] != "gen" or m["key"].split("|")[0] != "footstep":
            continue
        s = m["key"].split("|")[1]
        pred = np.array([SURFACES[int(i)] for i in mpred[g]])
        c = ctl.get(m["win"], {}).get(s)
        gen[g] = {"surface": s, "accuracy": frac(pred == s), "confusions": Counter(pred).most_common(),
                  "judged": bool(c and c["valid"]),
                  "ratio": frac(pred == s) / c["real"] if c and c["valid"] and c["real"] > 0 else None,
                  "reason": None if c and c["valid"] else ("no real %s single steps" % s if s not in built else
                                                           "CLAP fails the real %s control at %s s" % (s, m["win"]))}
    return ctl, gen


def shape_rows(clips):
    rows = []
    n5, n10 = max(1, int(0.005 * SR)), int(0.01 * SR)
    for x in clips:
        x = np.asarray(x, dtype=np.float64)
        env = np.sqrt(np.convolve(x * x, np.ones(n5) / n5, mode="same")) + 1e-12
        pk = int(np.argmax(env))
        db = 20.0 * np.log10(env / env[pk])
        above = np.flatnonzero(db[pk:] >= -30.0)
        j = pk + int(0.1 * SR)
        P = np.abs(np.fft.rfft(x)) ** 2
        f = np.fft.rfftfreq(len(x), 1.0 / SR)
        fr = x[: len(x) // n10 * n10].reshape(-1, n10)
        ms = np.mean(fr * fr, axis=1)
        fdb = 10.0 * np.log10(ms[ms > 1e-12]) if np.any(ms > 1e-12) else None
        rows.append({"tail_ms": float(above[-1] / SR * 1000.0) if len(above) else 0.0,
                     "decay100_db": float(np.median(db[j - n5: j + n5])) if j + n5 <= len(db) else NAN,
                     "low500": float(P[f < 500.0].sum() / max(P.sum(), 1e-20)),
                     "crest_db": float(crest_db(x.astype(np.float32))), "attack_ms": float(attack_ms(x.astype(np.float32))),
                     "floor_db": float(np.percentile(fdb, 10) - fdb.max()) if fdb is not None else NAN})
    return rows


def shape_compare(gen_rows, real_rows):
    out = {}
    for k in SHAPE:
        r = np.array([x[k] for x in real_rows], dtype=float)
        g = np.array([x[k] for x in gen_rows], dtype=float)
        r, g = r[np.isfinite(r)], g[np.isfinite(g)]
        if not r.size or not g.size:
            out[k] = {"real_median": NAN, "gen_median": NAN, "in_range": NAN}
            continue
        lo, hi = np.percentile(r, [10, 90])
        out[k] = {"real_median": float(np.median(r)), "real_p10": float(lo), "real_p90": float(hi),
                  "gen_median": float(np.median(g)), "in_range": float(np.mean((g >= lo) & (g <= hi)))}
    out["in_range_mean"] = frac([out[k]["in_range"] for k in SHAPE if out[k]["in_range"] == out[k]["in_range"]])
    return out


def shape_refs(meta, shapes, man, t, w, surface):
    pos = pos_sets(meta, t, w)
    rows = []
    for n in pos:
        for r, s in zip(shapes[n], meta[n]["surface"]):
            if t == "footstep" and surface in man.get("surfaces", {}) and s != surface:
                continue
            rows.append(r)
    src = "%s %s" % (t, surface if t == "footstep" and surface in man.get("surfaces", {}) else "all")
    return rows, src


def shape_strata(meta, shapes, man, t, w, surface):
    out = defaultdict(list)
    for n in pos_sets(meta, t, w):
        for r, s, st in zip(shapes[n], meta[n]["surface"], meta[n]["strata"]):
            if t == "footstep" and surface in man.get("surfaces", {}) and s != surface:
                continue
            out["%s:%s" % (meta[n]["set"], st)].append(r)
    return out


def routes_of(cfgs):
    return [s for s, c in cfgs.items() if "route" in c]


def shown(row, cfgs):
    return dedupe(([CURRENT] if CURRENT in cfgs else []) + ([row["best"]] if row["best"] else []) + routes_of(cfgs))


def summarize(cons, mat_gen, shapes_cmp, ab, chosen, man, args, cfgs, have):
    out = {}
    for et, mat, t in KEYS:
        k = key_name(et, mat)
        row = {"type": t, "settings": {}, "best": None, "reaches": False, "firm_reach": False, "flags": []}
        for s in cfgs:
            g = gen_name(s, et, mat)
            c = cons.get(g) or {"cons": None, "reason": "no real reference" if t is None else
                                "not generated by this route" if g not in have else "not scored"}
            row["settings"][s] = {"cons": c.get("cons"), "cons_ci": c.get("cons_ci"), "mean_ratio": c.get("mean_ratio"),
                                  "judges": {f: j["variant"] for f, j in c.get("judges", {}).items()},
                                  "reason": c.get("reason"), "shape_in_range": shapes_cmp.get(g, {}).get("in_range_mean"),
                                  "pq": ab.get(g, {}).get("gen_PQ"), "pq_real": ab.get(g, {}).get("real_PQ"),
                                  "material": mat_gen.get(g, {}).get("accuracy"),
                                  "material_ratio": mat_gen.get(g, {}).get("ratio"),
                                  "material_judged": mat_gen.get(g, {}).get("judged")}
        judged = [(s, v) for s, v in row["settings"].items() if v["cons"] is not None]
        if judged:
            best = max(judged, key=lambda sv: (sv[1]["cons"], sv[1]["mean_ratio"], sv[0] == CURRENT))
            row["best"] = best[0]
            row["reaches"] = bool(any(v["cons"] >= args.target for _s, v in judged))
            row["firm_reach"] = bool(any(v["cons_ci"][0] >= args.target for _s, v in judged))
        if t is None:
            row["flags"].append("no real reference")
        else:
            if man["types"][t].get("approx"):
                row["flags"].append("approximate reference")
            for w in WINS:
                if not chosen.get("%s@%s" % (t, w)):
                    row["flags"].append("no valid judge at %s s" % w)
            rn, rk = man["types"][t].get("clips", 0), man["types"][t].get("sources", 0)
            if rn < args.min_ref_clips or rk < args.min_ref_sources:
                row["flags"].append("too few real references for a judge (%d clips, %d sources)" % (rn, rk))
        if et == "footstep" and mat not in man.get("surfaces", {}):
            row["flags"].append("no real %s steps for the material check" % mat)
        out[k] = row
    return out


def mcnemar_pair(gen, cons, k, s, cfgs):
    et, mat = k.split("|")
    cur, bst = gen_name(CURRENT, et, mat), gen_name(s, et, mat)
    out = {}
    for f, j in cons.get(bst, {}).get("judges", {}).items():
        jc = cons.get(cur, {}).get("judges", {}).get(f)
        if not jc or jc["variant"] != j["variant"]:
            out[f] = None
            continue
        a = np.array(gen[bst][j["variant"]]["hits"], dtype=bool)
        b = np.array(gen[cur][j["variant"]]["hits"], dtype=bool)
        gn, ls = int((a & ~b).sum()), int((~a & b).sum())
        out[f] = {"variant": j["variant"], "gain": gn, "loss": ls, "p": mcnemar(gn, ls),
                  "cross_window": cfgs[s]["duration_s"] != cfgs[CURRENT]["duration_s"]}
    return out


def paired_vs_current(gen, cons, summary, cfgs):
    routes = routes_of(cfgs)
    for k, row in summary.items():
        row["mcnemar_vs_current"] = {}
        if routes:
            row["routes_vs_current"] = {s: mcnemar_pair(gen, cons, k, s, cfgs) if CURRENT in cfgs else {}
                                        for s in routes}
        if not row["best"] or row["best"] == CURRENT or CURRENT not in cfgs:
            continue
        row["mcnemar_vs_current"] = mcnemar_pair(gen, cons, k, row["best"], cfgs)


def build_listen(sets, summary, args, cfgs):
    gap, kgap = np.zeros(int(0.25 * SR), np.float32), np.zeros(int(0.8 * SR), np.float32)
    seq, order = [], []
    for et, mat, _t in KEYS:
        k = key_name(et, mat)
        best = summary[k]["best"]
        played = [s for s in shown(summary[k], cfgs) if gen_name(s, et, mat) in sets]
        for s in played:
            for x in sets[gen_name(s, et, mat)][: args.listen_seeds]:
                seq += [x, gap]
            order.append("%s %s" % (k, s))
        if not played:
            continue
        if not best:
            order[-1] += " (no judged setting)"
        seq.append(kgap)
    y = np.concatenate(seq)
    y = y / max(1e-9, float(np.abs(y).max())) * 0.9
    p = os.path.join(ROOT, args.listen)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    wavfile.write(p, SR, (y * 32767).astype(np.int16))
    return p, order


def ab_summary(ab, meta):
    per = {n: {a: float(np.mean(v[:, i])) for i, a in enumerate(AB_AXES)} for n, v in ab.items() if len(v)}
    out = {}
    for g, m in meta.items():
        if m["kind"] != "gen" or g not in per:
            continue
        r = {"gen_%s" % a: per[g][a] for a in AB_AXES}
        if m["type"]:
            pos = [n for n in pos_sets(meta, m["type"], m["win"]) if n in ab]
            if pos:
                v = np.concatenate([ab[n] for n in pos])
                r.update({"real_%s" % a: float(np.mean(v[:, i])) for i, a in enumerate(AB_AXES)})
        out[g] = r
    return per, out


def cell(c):
    if c is None or c.get("cons") is None:
        return "%-24s" % "  -- no judge"
    lo, hi = c["cons_ci"]
    return "%-24s" % ("%4.2f [%4.2f,%4.2f] %s" % (c["cons"], lo, hi, "".join(f[0] for f in sorted(c["judges"]))))


def report(meta, ctl, chosen, cons, gen, summary, mat_ctl, mat_gen, shapes_cmp, ab_per, labelsets, pools, runs,
           order, listen, args, types, family, man, cfgs):
    W = 168
    print()
    print("=" * W)
    print(" VOCABULARY JUDGE PANEL   positive control per type = real recordings cut like generated clips (_extract to the"
          " kept length, normalize_loudness at target_for('%s'))" % INTENSITY)
    print(" negatives = every real set of another family, up to %d clips per family (%d families), plus %d synthetic hiss"
          " clips; valid = real >= %.2f and real - pooled neg >= %.2f; usable = valid and real - hiss >= %.2f" % (
              args.neg_per_family, len(pools[WINS[0]]), len(meta["hiss@%s" % WINS[0]]["ids"]), args.min_real,
              args.min_gap, args.min_gap))
    print(" intervals: 95%% from %d bootstrap draws over independent sources (guns, uploaders, ESC-50 source files, HEAR"
          " shots) for real clips and over seeds for generated clips; F = usable with both lower bounds past the"
          " thresholds" % args.boot)
    print("=" * W)
    for w in WINS:
        print(" VALIDITY at %s s   per type: usable variants / total, then the chosen variant per model family" % w)
        for t in types:
            vs = [v for v in ctl if v.split("|")[2] == t and w in ctl[v]]
            us = [v for v in vs if ctl[v][w]["usable"]]
            ch = chosen.get("%s@%s" % (t, w), {})
            tag = "%s%s" % (t, " (approx)" if man["types"][t].get("approx") else "")
            if not ch:
                best = max(vs, key=lambda v: (ctl[v][w]["real"] - ctl[v][w]["neg"], ctl[v][w]["real"])) if vs else None
                b = ctl[best][w] if best else None
                few = b and (b["n_real"] < args.min_ref_clips or b["k_real"] < args.min_ref_sources)
                print("   %-24s usable %3d/%-3d  NO VALID JUDGE%s%s" % (
                    tag, len(us), len(vs), "   closest %s real %s neg %s hiss %s" % (
                        best, pct(b["real"]), pct(b["neg"]), pct(b["hiss"])) if b else "",
                    "   (%d clips from %d sources, under --min-ref-clips %d or --min-ref-sources %d)" % (
                        b["n_real"], b["k_real"], args.min_ref_clips, args.min_ref_sources) if few else ""))
                continue
            first = True
            for f, v in ch.items():
                c = ctl[v][w]
                print("   %-24s usable %3d/%-3d  %-5s %-46s real %s [%s,%s] neg %s hiss %s gap [%+.2f,%+.2f] AUC %.2f"
                      " worst %s %s %s%s" % (
                          tag if first else "", len(us), len(vs), f, v.split("|", 1)[1], pct(c["real"]),
                          pct(c["real_ci"][0]).strip(), pct(c["real_ci"][1]).strip(), pct(c["neg"]), pct(c["hiss"]),
                          c["gap_ci"][0], c["gap_ci"][1], c["auc"]["pooled"], c["worst_family"],
                          pct(c["neg_by_family"].get(c["worst_family"], NAN)), "F" if c["firm"] else "",
                          " strict" if c["strict"] else ""))
                first = False
        print("-" * W)
    print(" CLAP IDENTITY on real clips (one canonical phrase per type + %d extras + %d generic distractors), top-1 at 0.6 s"
          % (len(EXTRA), len(GENERIC)))
    for t in types:
        v = "clap|native|%s|id|top" % t
        if v in ctl and "0.6" in ctl[v]:
            c = ctl[v]["0.6"]
            print("   %-16s exact %s  family %s   top-1 %s" % (
                t, pct(c["real"]), pct(ctl.get("clap|native|%s|idfam|top" % t, {}).get("0.6", {}).get("real", NAN)),
                ", ".join("%s %d" % x for x in c["real_top1"])))
    print("=" * W)
    print(" CLAP MATERIAL CONTROL on real FoleySet single steps (5-way: %s; chance %.0f%%)" % (
        ", ".join(SURFACES), 100.0 / len(SURFACES)))
    for w, by in mat_ctl.items():
        for s, c in by.items():
            if s == "_all":
                print("   %s s  overall accuracy %s on %d clips, macro over built surfaces %s" % (
                    w, pct(c["accuracy"]), c["n"], pct(c["macro"])))
                continue
            print("   %s s  %-7s real %s [%s,%s] (%d clips, %d uploaders)  other surfaces called %s %s  %s   %s" % (
                w, s, pct(c["real"]), pct(c["real_ci"][0]).strip(), pct(c["real_ci"][1]).strip(), c["n"], c["k"], s,
                pct(c["neg"]), "valid" if c["valid"] else "INVALID", ", ".join("%s %d" % x for x in c["confusions"])))
    for s in SURFACES:
        if s not in man.get("surfaces", {}):
            print("   %-7s no real reference (%s)" % (s, man.get("surfaces_dropped", {}).get(s, {}).get("reason", "")))
    print("=" * W)
    print(" KEYS   cons = min over chosen judges of min(rate / real rate, 1) [95%% interval]; judges a=ast p=panns c=clap;"
          " target %.2f of real" % args.target)
    print(" %-19s %-15s " % ("key", "type") + " ".join("%-24s" % s for s in cfgs) +
          " %-13s %-6s %-6s %s" % ("best", ">=%.1f" % args.target, "firm", "flags"))
    for k, row in summary.items():
        et, mat = k.split("|")
        cells = [cell(cons.get(gen_name(s, et, mat))) for s in cfgs]
        print(" %-19s %-15s " % (k, row["type"] or "-") + " ".join(cells) + " %-13s %-6s %-6s %s" % (
            row["best"] or "-", "yes" if row["reaches"] else "no", "yes" if row["firm_reach"] else "no",
            "; ".join(row["flags"])))
    print("-" * W)
    print(" PER JUDGE rate (ratio to real) for the current and best settings, with McNemar best vs current paired on seed")
    for k, row in summary.items():
        et, mat = k.split("|")
        for s in shown(row, cfgs):
            c = cons.get(gen_name(s, et, mat), {})
            if not c.get("judges"):
                continue
            parts = []
            for f, j in c["judges"].items():
                mc = None if s == CURRENT else row["mcnemar_vs_current"].get(f) if s == row["best"] else \
                    row.get("routes_vs_current", {}).get(s, {}).get(f)
                parts.append("%s %s (%.2f [%.2f,%.2f])%s" % (
                    f, pct(j["rate"]), j["ratio"], j["ratio_ci"][0], j["ratio_ci"][1],
                    " +%d/-%d p%.2f%s" % (mc["gain"], mc["loss"], mc["p"], "x" if mc["cross_window"] else "")
                    if mc else ""))
            print("   %-19s %-13s %s" % (k, s, "   ".join(parts)))
    print("-" * W)
    print(" UNVALIDATED keys, CLAP identity top-1 shown for information only")
    for et, mat, t in KEYS:
        if t is not None:
            continue
        for s in cfgs:
            g = gen_name(s, et, mat)
            top = gen.get("_identity", {}).get(g)
            print("   %-19s %-13s %s" % (key_name(et, mat), s, ", ".join("%s %d" % x for x in top) if top else "-"))
    print("-" * W)
    print(" FOOTSTEP MATERIAL on generated clips (CLAP 5-way; judged only where the real control for that surface passes)")
    for et, mat, _t in KEYS:
        if et != "footstep":
            continue
        cells = []
        for s in cfgs:
            r = mat_gen.get(gen_name(s, et, mat))
            cells.append("%s %s%s" % (s, pct(r["accuracy"]) if r else "  nan",
                                      " (%.2f of real)" % r["ratio"] if r and r["ratio"] is not None else "*"))
        print("   %-19s %s" % (key_name(et, mat), "   ".join(cells)))
    print("   * = unjudged (%s)" % "; ".join(sorted({r["reason"] for r in mat_gen.values() if r["reason"]})))
    print("-" * W)
    print(" SHAPE vs the real set at the same length (gen median / real median, share of generated clips inside the real"
          " 10-90% range) and Audiobox PQ gen / real")
    for k, row in summary.items():
        et, mat = k.split("|")
        for s in shown(row, cfgs):
            g = gen_name(s, et, mat)
            if g not in meta:
                continue
            sc = shapes_cmp.get(g)
            pq = ab_per.get(g, {}).get("PQ")
            rpq = row["settings"][s]["pq_real"]
            if not sc:
                print("   %-19s %-13s no real set for shape   PQ %s" % (k, s, "%.2f" % pq if pq is not None else "-"))
                continue
            print("   %-19s %-13s %s  in-range %.2f  PQ %s / %s  ref %s" % (
                k, s, "  ".join("%s %.3g/%.3g %.0f%%" % (x, sc[x]["gen_median"], sc[x]["real_median"],
                                                          100 * sc[x]["in_range"]) for x in SHAPE),
                sc["in_range_mean"], "%.2f" % pq if pq is not None else "-", "%.2f" % rpq if rpq is not None else "-",
                sc.get("_ref", "")))
            for st, v in sc.get("by_stratum", {}).items():
                print("   %-19s %-13s   %-30s %4d real  %s  in-range %.2f" % (
                    "", "", st, v["n"], "  ".join("%s %.3g/%.3g %.0f%%" % (
                        x, v[x]["gen_median"], v[x]["real_median"], 100 * v[x]["in_range"]) for x in SHAPE),
                    v["in_range_mean"]))
    print("-" * W)
    print(" GENERATION  median ms per clip and static guard counts per setting")
    for s in cfgs:
        if "route" in cfgs[s]:
            print("   %-13s route %s, %d keys, steps %s, route.json %s" % (
                s, cfgs[s]["route"], cfgs[s]["keys"], cfgs[s]["steps"], "yes" if cfgs[s]["route_json"] else "no"))
            continue
        rs = [runs[gen_name(s, et, mat)] for et, mat, _t in KEYS]
        ms = [r["ms_median"] for r in rs if r["ms_median"] is not None]
        st = Counter()
        for r in rs:
            st.update(r["static"])
        print("   %-13s median %s ms  %s" % (s, "%.1f" % float(np.median(ms)) if ms else "reused",
                                             "  ".join("%s %d" % x for x in sorted(st.items()))))
    print("=" * W)
    nv = {w: [t for t in types if not chosen.get("%s@%s" % (t, w))] for w in WINS}
    for w in WINS:
        print(" NO VALID JUDGE at %s s: %s" % (w, ", ".join(nv[w]) or "none"))
    print(" NO REAL REFERENCE: %s" % ", ".join(key_name(et, mat) for et, mat, t in KEYS if t is None))
    print(" APPROXIMATE REFERENCE: %s" % ", ".join("%s (%s)" % (t, man["types"][t]["note"]) for t in types
                                                if man["types"][t].get("approx")))
    print(" LISTEN %s in order: %s" % (listen, " | ".join(order)))
    print("=" * W)


def parse(argv):
    ap = argparse.ArgumentParser(description="Validate judges per sound type on real references, then score SARGE's new"
                                             " vocabulary under four generic settings.")
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--refs", default=None)
    ap.add_argument("--judges", nargs="+", default=["ast", "panns", "clap", "audiobox"])
    ap.add_argument("--tag-fits", nargs="+", default=TAG_FITS, choices=TAG_FITS)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--reuse", action="store_true")
    ap.add_argument("--gen-dir", default=os.path.join("data", "_gate", "vocab"))
    ap.add_argument("--settings", nargs="+", default=list(SETTINGS), choices=list(SETTINGS) + ["none"])
    ap.add_argument("--route", action="append", default=[])
    ap.add_argument("--strata-shape", action="store_true")
    ap.add_argument("--min-stratum", type=int, default=8)
    ap.add_argument("--listen", default=None)
    ap.add_argument("--listen-seeds", type=int, default=3)
    ap.add_argument("--fpr", type=float, default=0.1)
    ap.add_argument("--min-real", type=float, default=0.5)
    ap.add_argument("--min-gap", type=float, default=0.3)
    ap.add_argument("--target", type=float, default=0.8)
    ap.add_argument("--min-ref-clips", type=int, default=10)
    ap.add_argument("--min-ref-sources", type=int, default=4)
    ap.add_argument("--neg-per-family", type=int, default=40)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--boot-seed", type=int, default=0)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    bad = [j for j in args.judges if j not in JUDGES]
    if bad:
        raise SystemExit("unknown judges %s, choose from %s" % (bad, sorted(JUDGES)))
    if args.refs:
        if args.manifest != MANIFEST:
            raise SystemExit("--refs and --manifest both given, pick one")
        args.manifest = os.path.join(args.refs, "manifest.json")
    if "none" in args.settings:
        if len(args.settings) > 1:
            raise SystemExit("--settings none takes no other setting")
        args.settings = []
    if not args.settings and not args.route:
        raise SystemExit("nothing to score: --settings none and no --route")
    names = []
    for spec in args.route:
        name, _, d = spec.partition("=")
        if not ROUTE_NAME.fullmatch(name) or not d:
            raise SystemExit("--route takes NAME=DIR with NAME of letters, digits and _ . + -, got %r" % spec)
        if name in SETTINGS or name in names:
            raise SystemExit("route name %s is already a setting or another route" % name)
        names.append(name)
    parts = []
    if os.path.normcase(os.path.normpath(args.manifest)) != os.path.normcase(os.path.normpath(MANIFEST)):
        parts.append(os.path.basename(os.path.dirname(os.path.normpath(os.path.join(ROOT, args.manifest)))))
    if args.settings != list(SETTINGS):
        parts.append("+".join(args.settings) or "nobase")
    tag = "".join("_" + p for p in parts + names)
    if args.out is None:
        args.out = os.path.join(ROOT, "scripts", "bench", "out_vocab_eval%s.json" % tag)
    if args.listen is None:
        args.listen = os.path.join("data", "_gate", "listen", "vocab_best%s.wav" % tag)
    return args


def main(argv=None):
    args = parse(argv)
    t0 = time.perf_counter()
    boot = Boot(args.boot, args.boot_seed)
    base = {s: SETTINGS[s] for s in args.settings}
    man, sets, meta = load_refs(args, base)
    rcfgs, rsets, rmeta, rruns = load_routes(args, man)
    cfgs = {**base, **rcfgs}
    build_hiss(sets, meta)
    types = list(man["types"])
    family = {t: man["types"][t]["family"] for t in types}
    print("references: %d sets, %d clips (%.1f s)" % (len(sets), sum(len(v) for v in sets.values()),
                                                     time.perf_counter() - t0), flush=True)
    gsets, gmeta, runs, ginfo = generate(args, base)
    sets.update(gsets)
    meta.update(gmeta)
    sets.update(rsets)
    meta.update(rmeta)
    runs.update(rruns)
    pools = neg_pools(meta, args.neg_per_family)
    var, labelsets, mpred, ab = {}, {}, {}, {}
    for j in args.judges:
        t1 = time.perf_counter()
        res = JUDGES[j](sets, args)
        if j == "clap":
            v, ls = clap_variants(res, types, family)
            mpred = material_preds(res)
        elif j == "audiobox":
            v, ls = {}, {}
            ab = res["scores"]
        else:
            v, ls = tagger_variants(j, res, types)
        var.update(v)
        labelsets[j] = ls
        del res
        free_gpu(j)
        print("judged %-8s %.1f s" % (j, time.perf_counter() - t1), flush=True)
    ctl, taus = evaluate(var, meta, pools, family, args, boot)
    chosen = choose(ctl, types)
    gen, cons = score_gen(var, ctl, taus, meta, chosen, pools, family, args, boot)
    idv = next((v for v in var if v.endswith("|id")), None)
    gen["_identity"] = {g: Counter(var[idv][g]["top1"]).most_common(3) for g, m in meta.items()
                        if m["kind"] == "gen"} if idv else {}
    mat_ctl, mat_gen = material(mpred, meta, man, args, boot) if mpred else ({}, {})
    shapes = {n: shape_rows(x) for n, x in sets.items() if not n.startswith("hiss@")}
    shapes_cmp = {}
    for g, m in meta.items():
        if m["kind"] != "gen" or not m["type"]:
            continue
        rows, src = shape_refs(meta, shapes, man, m["type"], m["win"], m["key"].split("|")[1])
        if rows:
            shapes_cmp[g] = {**shape_compare(shapes[g], rows), "_ref": src}
            if args.strata_shape:
                by = shape_strata(meta, shapes, man, m["type"], m["win"], m["key"].split("|")[1])
                shapes_cmp[g]["by_stratum"] = {k: {**shape_compare(shapes[g], v), "n": len(v)}
                                               for k, v in sorted(by.items()) if len(v) >= args.min_stratum}
    ab_per, ab_gen = ab_summary(ab, meta) if ab else ({}, {})
    summary = summarize(cons, mat_gen, shapes_cmp, ab_gen, chosen, man, args, cfgs,
                        {g for g, m in meta.items() if m["kind"] == "gen"})
    paired_vs_current(gen, cons, summary, cfgs)
    listen, order = build_listen(sets, summary, args, cfgs)
    report(meta, ctl, chosen, cons, gen, summary, mat_ctl, mat_gen, shapes_cmp, ab_per, labelsets, pools, runs, order,
           rel(listen), args, types, family, man, cfgs)
    out = {"meta": {"root": ROOT, "args": vars(args), "sr": SR, "intensity": INTENSITY, "settings": cfgs,
                    "current": CURRENT, "production": PRODUCTION, "generator": ginfo,
                    "keys": [{"key": key_name(et, mat), "type": t} for et, mat, t in KEYS],
                    "rules": list(RULES), "tag_fits": args.tag_fits, "clap_texts": CLAP_TEXTS, "phrases": PHRASES,
                    "canon": CANON, "extra": EXTRA, "same_event": SAME_EVENT, "generic": GENERIC,
                    "material_text": MAT_TEXT,
                    "hiss": {"spectra_slope_highpass": HISS_SPECTRA, "per_spectrum": HISS_PER},
                    "manifest": args.manifest, "heldout": man["meta"].get("heldout"),
                    "seconds": time.perf_counter() - t0},
           "label_sets": labelsets, "types": man["types"], "families": family,
           "neg_pools": {w: {f: {"n": p["n"], "sets": p["sets"], "sources": len(set(p["clusters"]))}
                             for f, p in by.items()} for w, by in pools.items()},
           "sets": {n: {**{k: v for k, v in m.items() if k not in ("ids", "clusters", "strata", "surface")},
                        "n": len(sets[n]), "sources": len(set(m["clusters"]))} for n, m in meta.items()},
           "validity": ctl, "chosen": chosen, "generated": gen, "consensus": cons, "summary": summary,
           "material": {"control": mat_ctl, "generated": mat_gen}, "shape": shapes_cmp,
           "audiobox": {"per_set": ab_per, "generated": ab_gen}, "runs": runs,
           "listen": {"path": rel(listen), "order": order}}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print("saved -> %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

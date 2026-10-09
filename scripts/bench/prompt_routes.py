import os
import sys

BENCH = os.path.join(os.getcwd(), "scripts", "bench")
if not os.path.isfile(os.path.join(BENCH, "_bootstrap.py")):
    BENCH = r"D:\Python\SAGE\scripts\bench"
sys.path.insert(0, BENCH)

import _bootstrap

import argparse
import json
import time

import numpy as np

import vocab_eval as ve
from sarge.generators.stable_audio import _FOOTSTEP_PLAIN, _IMPACT_EVENTS, _IMPACT_TAIL, _MATERIAL_ADJ, _SPLASHY, \
    build_prompt
from sarge.generators.stable_audio3 import StableAudio3Generator
from sarge.types import CacheKey

ROOT = os.path.dirname(os.path.dirname(BENCH))
OUT = os.path.join("data", "_gate", "prompt_routes")
SUFFIX = ", dry, isolated sound effect"


def compound(event, material, intensity="normal"):
    lead = "" if intensity in (None, "", "normal") else f"{intensity} "
    if event in ("splash", "drip", "pour") or material in _SPLASHY or event == "footstep":
        return build_prompt(event, material, intensity)
    if event in _IMPACT_EVENTS and _IMPACT_TAIL.get(material):
        return build_prompt(event, material, intensity)
    head = f"{lead}{material.replace('_', ' ')} {event.replace('_', ' ')}"
    return ("an " if head[:1] in "aeiou" else "a ") + head + SUFFIX


def nodry(fn):
    return lambda event, material, intensity="normal": fn(event, material, intensity).replace(
        SUFFIX, ", isolated sound effect")


VARIANTS = {
    "compound": compound,
    "nodry": nodry(build_prompt),
    "compound_nodry": nodry(compound),
}


def parse(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--out-dir", default=OUT)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse(argv)
    cfg = ve.SETTINGS[ve.CURRENT]
    g = StableAudio3Generator(sampler_type="euler", steps=cfg["steps"], detail_steps=cfg["detail_steps"],
                              gen_duration_s=cfg["gen_duration_s"], duration_s=cfg["duration_s"])
    g.ensure_ready()
    keys = [(et, mat) for et, mat, _t in ve.KEYS]
    g.generate(CacheKey(keys[0][0], keys[0][1], ve.INTENSITY), variant=0)
    report = {}
    for name in args.variants:
        fn = VARIANTS[name]
        d = os.path.join(ROOT, args.out_dir, name)
        os.makedirs(d, exist_ok=True)
        prompts = {"%s|%s" % k: fn(k[0], k[1], ve.INTENSITY) for k in keys}
        g.prompts = {str(CacheKey(et, mat, ve.INTENSITY)): prompts["%s|%s" % (et, mat)] for et, mat in keys}
        r0 = dict(g.static_stats)
        ms, t0 = [], time.perf_counter()
        for et, mat in keys:
            k = CacheKey(et, mat, ve.INTENSITY)
            for v in range(args.seeds):
                t1 = time.perf_counter()
                c = g.generate(k, variant=v)
                ms.append((time.perf_counter() - t1) * 1000.0)
                c.to_wav(os.path.join(d, "%s-%s_%02d.wav" % (et, mat, v)))
        static = {a: int(g.static_stats.get(a, 0)) - int(r0.get(a, 0)) for a in g.static_stats}
        changed = sorted(k for k, p in prompts.items() if p != build_prompt(*k.split("|"), ve.INTENSITY))
        json.dump({"steps": cfg["steps"], "prompts": prompts, "clusters": [], "sources": [], "audio": "none",
                   "note": "base small-sfx at the production setting with prompt variant %s; no real audio" % name,
                   "changed_keys": changed, "static": static},
                  open(os.path.join(d, "route.json"), "w", encoding="utf-8"), indent=1)
        report[name] = {"ms_median": float(np.median(ms)), "seconds": time.perf_counter() - t0, "static": static,
                        "changed_keys": changed}
        print("variant %-15s %d keys changed  median %.1f ms  %s" % (
            name, len(changed), report[name]["ms_median"], " ".join("%s %d" % kv for kv in sorted(static.items()))),
            flush=True)
        for k in keys:
            kn = "%s|%s" % k
            print("   %-20s %s" % (kn, prompts[kn]), flush=True)
    g.prompts = {}
    print("routes -> %s  (score with vocab_eval.py --route NAME=%s/NAME)" % (args.out_dir, args.out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

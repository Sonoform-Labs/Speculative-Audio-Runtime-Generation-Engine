import _bootstrap

import argparse
import gc
import json
import os
import time

import numpy as np
import torch

from sarge.generators.stable_audio3 import StableAudio3Generator
from sarge.types import CacheKey
from validate_panel import CATEGORIES, _resample
from gate_quality import crest_db, attack_ms, onset_window
from ab_clap import clap_scores, mcnemar
from score_hard import HARD

MATS = ["wood", "stone", "grass", "metal"]
FOOT_CONFIGS = {
    "s3_euler_cfg2": dict(steps=3, sampler_type="euler", cfg_scale=2.0),
    "s4_euler_cfg2": dict(steps=4, sampler_type="euler", cfg_scale=2.0),
    "s6_euler_cfg2": dict(steps=6, sampler_type="euler", cfg_scale=2.0),
    "s8_euler_cfg2": dict(steps=8, sampler_type="euler", cfg_scale=2.0),
    "s4_pingpong_cfg2": dict(steps=4, sampler_type="pingpong", cfg_scale=2.0),
    "s8_pingpong_cfg2": dict(steps=8, sampler_type="pingpong", cfg_scale=2.0),
    "s8_pingpong_cfg1": dict(steps=8, sampler_type="pingpong", cfg_scale=1.0),
}
WATER_KEYS = [CacheKey("splash", "water", "normal"), CacheKey("footstep", "water", "normal"),
              CacheKey("impact", "water", "heavy"), CacheKey("land", "water", "normal")]
WATER_STEPS = [3, 4, 8]
WATER_TEXTS = ["water splashing", "a footstep", "a knock", "an explosion", "rain", "hands clapping",
               "a ball bouncing", "a door closing", "music", "speech"]


def clip_rows(g, keys, variants, out_dir, tag):
    rows, clips = [], []
    os.makedirs(out_dir, exist_ok=True)
    g.generate(keys[0], variant=0)
    ts = []
    for k in keys:
        for v in range(variants):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            c = g.generate(k, variant=v)
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000.0)
            p = os.path.join(out_dir, "%s-%s_%02d.wav" % (k.event_type, k.material, v))
            c.to_wav(p)
            w = onset_window(c.samples, c.sample_rate)
            rows.append({"cfg": tag, "key": "%s|%s" % (k.event_type, k.material), "mat": k.material, "v": v,
                         "path": p, "crest": crest_db(w), "attack": attack_ms(w, c.sample_rate)})
            clips.append(_resample(c.samples, c.sample_rate, 48000))
    return rows, clips, float(np.median(ts))


def paired(rows, ref, f):
    R = {(r["key"], r["v"]): r for r in rows if r["cfg"] == ref}
    out = {}
    for cfg in sorted({r["cfg"] for r in rows}):
        A = {(r["key"], r["v"]): r for r in rows if r["cfg"] == cfg}
        ks = [k for k in R if k in A]
        a = np.array([A[k][f] for k in ks])
        b = np.array([R[k][f] for k in ks])
        g, l = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
        out[cfg] = (float(a.mean()), g, l, mcnemar(g, l))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", type=int, default=10)
    ap.add_argument("--water-variants", type=int, default=10)
    ap.add_argument("--out", default=os.path.join("scripts", "bench", "out_tier_probe.json"))
    args = ap.parse_args()

    g = StableAudio3Generator()
    g.ensure_ready()
    base = dict(steps=g.steps, sampler_type=g.sampler_type, cfg_scale=g.cfg_scale, detail_steps=g.detail_steps)
    foot_keys = [CacheKey("footstep", m, "normal") for m in MATS]
    frows, fclips, fms = [], [], {}
    for name, cfg in FOOT_CONFIGS.items():
        for a, v in base.items():
            setattr(g, a, v)
        for a, v in cfg.items():
            setattr(g, a, v)
        r, c, ms = clip_rows(g, foot_keys, args.variants, os.path.join("data", "_gate", "tier", name), name)
        frows += r
        fclips += c
        fms[name] = ms
        print("generated %-18s median %.1f ms" % (name, ms))
    wrows, wclips, wms = [], [], {}
    for s in WATER_STEPS:
        for a, v in base.items():
            setattr(g, a, v)
        g.detail_steps = s
        name = "water_s%d" % s
        r, c, ms = clip_rows(g, WATER_KEYS, args.water_variants, os.path.join("data", "_gate", "tier_water", name), name)
        wrows += r
        wclips += c
        wms[name] = ms
        print("generated %-18s median %.1f ms" % (name, ms))
    del g
    gc.collect()
    torch.cuda.empty_cache()

    sc = clap_scores({"f": fclips, "w": wclips},
                     {"base": CATEGORIES, "hard": HARD, "mat": ["the sound of a footstep on %s" % m for m in MATS],
                      "water": WATER_TEXTS}, "cuda", 32)
    fb = sc["f"]["base"].argmax(1) == CATEGORIES.index("a footstep")
    fh = sc["f"]["hard"].argmax(1) == HARD.index("a footstep")
    fm = sc["f"]["mat"].argmax(1) == np.array([MATS.index(r["mat"]) for r in frows])
    for i, r in enumerate(frows):
        r["base"], r["hard"], r["matok"] = int(fb[i]), int(fh[i]), int(fm[i])
    wtop = sc["w"]["water"].argmax(1)
    wsm = sc["w"]["water"] - sc["w"]["water"].max(1, keepdims=True)
    wp = np.exp(wsm) / np.exp(wsm).sum(1, keepdims=True)
    for i, r in enumerate(wrows):
        r["water_top"] = int(WATER_TEXTS[int(wtop[i])] == "water splashing")
        r["p_water"] = float(wp[i, 0])

    from audiobox_aesthetics.infer import initialize_predictor
    pred = initialize_predictor()
    allrows = frows + wrows
    for i in range(0, len(allrows), 40):
        chunk = allrows[i: i + 40]
        for r, o in zip(chunk, pred.forward([{"path": r["path"]} for r in chunk])):
            r["PQ"] = float(o["PQ"])
            r["CE"] = float(o["CE"])

    res = {"foot": {}, "water": {}}
    pb, ph, pm = paired(frows, "s3_euler_cfg2", "base"), paired(frows, "s3_euler_cfg2", "hard"), paired(frows, "s3_euler_cfg2", "matok")
    print()
    print(" FOOTSTEP TIERS (n=%d each, paired vs s3_euler_cfg2)" % (len(MATS) * args.variants))
    print("%-18s %7s %16s %16s %16s %7s %7s %6s" % ("config", "ms", "clap foot", "clap hard", "material", "crest", "attack", "PQ"))
    for name in FOOT_CONFIGS:
        rs = [r for r in frows if r["cfg"] == name]
        d = {"ms": fms[name], "base": pb[name], "hard": ph[name], "mat": pm[name],
             "crest": float(np.mean([r["crest"] for r in rs])), "attack": float(np.median([r["attack"] for r in rs])),
             "PQ": float(np.mean([r["PQ"] for r in rs]))}
        res["foot"][name] = d
        fmt = lambda t: "%3.0f%% +%d/-%d p%.2f" % (100 * t[0], t[1], t[2], t[3])
        print("%-18s %7.1f %16s %16s %16s %7.1f %7.2f %6.2f" % (name, d["ms"], fmt(d["base"]), fmt(d["hard"]), fmt(d["mat"]),
                                                              d["crest"], d["attack"], d["PQ"]))
    pw = paired(wrows, "water_s8", "water_top")
    print()
    print(" WATER DETAIL STEPS (n=%d each over %s, paired vs water_s8)" % (len(WATER_KEYS) * args.water_variants,
                                                                         ", ".join("%s|%s" % (k.event_type, k.material) for k in WATER_KEYS)))
    print("%-10s %7s %18s %9s %7s %6s %6s" % ("config", "ms", "clap water top1", "p_water", "crest", "PQ", "CE"))
    for name in wms:
        rs = [r for r in wrows if r["cfg"] == name]
        d = {"ms": wms[name], "water_top": pw[name], "p_water": float(np.mean([r["p_water"] for r in rs])),
             "crest": float(np.mean([r["crest"] for r in rs])), "PQ": float(np.mean([r["PQ"] for r in rs])),
             "CE": float(np.mean([r["CE"] for r in rs])),
             "PQ_by_key": {k: float(np.mean([r["PQ"] for r in rs if r["key"] == k])) for k in sorted({r["key"] for r in rs})}}
        res["water"][name] = d
        t = d["water_top"]
        print("%-10s %7.1f %4.0f%% +%d/-%d p%.2f %9.3f %7.1f %6.2f %6.2f" % (name, d["ms"], 100 * t[0], t[1], t[2], t[3],
                                                                          d["p_water"], d["crest"], d["PQ"], d["CE"]))
    json.dump({"result": res, "rows": allrows}, open(args.out, "w"), indent=1)
    print("saved -> %s" % args.out)
    print("listening folders: data\\_gate\\tier\\<config> and data\\_gate\\tier_water\\water_s3|s4|s8")


if __name__ == "__main__":
    main()

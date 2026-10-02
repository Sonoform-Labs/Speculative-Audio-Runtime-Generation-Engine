import _bootstrap

import argparse
import gc
import itertools
import json
import os
from math import sqrt

import numpy as np
import torch

from sarge.generators.stable_audio import _MATERIAL_ADJ, _seed
from sarge.generators.stable_audio3 import StableAudio3Generator
from sarge.types import AudioClip, CacheKey
from validate_panel import CATEGORIES, _resample
from gate_quality import transients, crest_db, attack_ms, onset_window
from ab_clap import gen_one, clap_scores, softmax, wilson, mcnemar

FACTORS = ["single", "normal", "adj", "clear", "closemic"]
FOOT_IDX = CATEGORIES.index("a footstep")


def compose(bits, mat):
    on = dict(zip(FACTORS, bits))
    head = "a "
    if on["single"]:
        head += "single "
    if on["normal"]:
        head += "normal "
    if on["adj"]:
        head += _MATERIAL_ADJ.get(mat, mat) + " "
    s = head + "footstep on " + mat
    if on["clear"]:
        s += ", a clear distinct footstep"
    if on["closemic"]:
        s += ", close mic"
    return s + ", dry, isolated sound effect"


def code(bits):
    return "".join(str(b) for b in bits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--sampler", default="euler")
    ap.add_argument("--variants", type=int, default=10)
    ap.add_argument("--materials", nargs="+", default=["wood", "stone", "grass", "metal"])
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--codes", default="")
    ap.add_argument("--v0", type=int, default=0)
    ap.add_argument("--tag", default="pf_")
    ap.add_argument("--audiobox", action="store_true")
    ap.add_argument("--out", default=os.path.join("scripts", "bench", "out_prompt_factorial.json"))
    args = ap.parse_args()
    dev = "cuda"
    mats = args.materials
    combos = list(itertools.product([0, 1], repeat=len(FACTORS)))
    if args.codes:
        combos = [tuple(int(ch) for ch in c) for c in args.codes.split(",")]

    g = StableAudio3Generator()
    g.ensure_ready()
    sr = int(g.sample_rate)
    gen_one(g, compose(combos[0], mats[0]), 2, args.sampler, 0)

    rows, c06 = [], []
    for bits in combos:
        tag = args.tag + code(bits)
        d = os.path.join("data", "_gate", tag)
        os.makedirs(d, exist_ok=True)
        ms = []
        for mat in mats:
            key = CacheKey("footstep", mat, "normal")
            text = compose(bits, mat)
            for v in range(args.v0, args.v0 + args.variants):
                x06, _x10, t = gen_one(g, text, args.steps, args.sampler, _seed(key, v))
                AudioClip(x06, sr).to_wav(os.path.join(d, "%s_%02d.wav" % (mat, v)))
                r = {"code": code(bits), "mat": mat, "v": v, "ms": t, "text": text}
                r.update(dict(zip(FACTORS, bits)))
                rows.append(r)
                c06.append(x06)
                ms.append(t)
        print("generated %s %3d clips median %.1f ms   %s" % (tag, len(ms), np.median(ms), compose(bits, "metal")))

    g._model = None
    del g
    gc.collect()
    torch.cuda.empty_cache()

    text_sets = {"mat": ["the sound of a footstep on %s" % x for x in mats], "cat": CATEGORIES}
    sc = clap_scores({"x06": [_resample(x, sr, 48000) for x in c06]}, text_sets, dev, args.bs)["x06"]
    true = np.array([mats.index(r["mat"]) for r in rows])
    pt = softmax(sc["mat"])[np.arange(len(rows)), true]
    pred = sc["mat"].argmax(1)
    foot = sc["cat"].argmax(1) == FOOT_IDX
    for i, r in enumerate(rows):
        w = onset_window(c06[i], sr)
        r["hit"] = int(pred[i] == true[i])
        r["pt"] = float(pt[i])
        r["foot"] = int(foot[i])
        r["crest"] = crest_db(w)
        r["attack"] = attack_ms(w, sr)

    if args.audiobox:
        from audiobox_aesthetics.infer import initialize_predictor
        pred_ab = initialize_predictor()
        for c in sorted({r["code"] for r in rows}):
            idx = [i for i, r in enumerate(rows) if r["code"] == c]
            paths = [os.path.join("data", "_gate", args.tag + c, "%s_%02d.wav" % (rows[i]["mat"], rows[i]["v"])) for i in idx]
            ab = pred_ab.forward([{"path": p} for p in paths])
            for i, o in zip(idx, ab):
                rows[i]["PQ"] = float(o["PQ"])
                rows[i]["CE"] = float(o["CE"])

    W = 132
    cells = []
    for bits in combos:
        c = code(bits)
        rs = [r for r in rows if r["code"] == c]
        n = len(rs)
        k = sum(r["hit"] for r in rs)
        f = sum(r["foot"] for r in rs)
        per = "/".join(str(sum(r["hit"] for r in rs if r["mat"] == m)) for m in mats)
        tr = transients([c06[i] for i, r in enumerate(rows) if r["code"] == c], sr)
        cells.append({"code": c, "n": n, "mat_acc": k / n, "mat_ci": wilson(k, n), "foot": f / n,
                      "foot_ci": wilson(f, n), "pt": float(np.mean([r["pt"] for r in rs])),
                      "both": float(np.mean([r["hit"] * r["foot"] for r in rs])),
                      "per_mat": per, "crest": tr["crest_db"], "attack": tr["attack_ms"],
                      "PQ": float(np.mean([r.get("PQ", float("nan")) for r in rs])),
                      "example": compose(bits, "metal")})
    cells.sort(key=lambda c: (-(c["mat_acc"] + c["foot"]), c["code"]))

    print()
    print("=" * W)
    print(" PROMPT FACTORIAL  footstep x %s, %s@%d, %d clips per prompt, bits = %s"
          % ("/".join(mats), args.sampler, args.steps, len(mats) * args.variants, " ".join(FACTORS)))
    print(" anchors: 11001 = June plain prompt, 00110 = shipped enriched prompt")
    print("=" * W)
    print("%-6s %8s %8s %8s %6s %7s %7s %6s  %-12s %s" % ("bits", "mat", "foot", "both", "pT", "crest", "attack", "PQ",
                                                       "hit w/s/g/m", "prompt for metal"))
    print("-" * W)
    for c in cells:
        print("%-6s %7.0f%% %7.0f%% %7.0f%% %6.3f %7.1f %7.2f %6.2f  %-12s %s" % (
            c["code"], 100 * c["mat_acc"], 100 * c["foot"], 100 * c["both"], c["pt"], c["crest"],
            c["attack"], c["PQ"], c["per_mat"], c["example"]))
    print("-" * W)
    print(" mat = CLAP material argmax over 'the sound of a footstep on X'; foot = CLAP picks 'a footstep' among %d categories;"
          % len(CATEGORIES))
    print(" both = clip passes both; real footsteps: crest 18.7 dB, attack 10.40 ms; CLAP footstep on real 0.6 s crops 95%")

    if args.codes:
        print()
        print(" PAIRED vs %s (same seed and material)" % args.codes.split(",")[0])
        ref = {(r["mat"], r["v"]): r for r in rows if r["code"] == args.codes.split(",")[0]}
        for c in args.codes.split(",")[1:]:
            oth = {(r["mat"], r["v"]): r for r in rows if r["code"] == c}
            ks = [k for k in ref if k in oth]
            for f in ("hit", "foot"):
                a = np.array([oth[k][f] for k in ks])
                b = np.array([ref[k][f] for k in ks])
                p = mcnemar(int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum()))
                print("  %s %-4s %5.0f%% vs %5.0f%%  gained %3d lost %3d  mcnemarP %.4f" % (
                    c, f, 100 * a.mean(), 100 * b.mean(), int(((a == 1) & (b == 0)).sum()),
                    int(((a == 0) & (b == 1)).sum()), p))
        print("=" * W)
        json.dump({"args": vars(args), "cells": cells, "rows": rows}, open(args.out, "w"), indent=1)
        print("saved -> %s" % args.out)
        return

    print()
    print(" MAIN EFFECTS (paired: same seed, material and the other four factors)")
    print("  %-10s %6s %15s %9s %15s %9s %8s %8s" % ("factor", "mat", "on-off", "mcnemarP", "foot on-off", "mcnemarP",
                                                    "dCrest", "dAttack"))
    effects = []
    for fi, fname in enumerate(FACTORS):
        idx_on = {(r["code"][:fi] + r["code"][fi + 1:], r["mat"], r["v"]): r for r in rows if r[fname] == 1}
        idx_off = {(r["code"][:fi] + r["code"][fi + 1:], r["mat"], r["v"]): r for r in rows if r[fname] == 0}
        ks = [k for k in idx_on if k in idx_off]
        a = np.array([idx_on[k]["hit"] for k in ks])
        b = np.array([idx_off[k]["hit"] for k in ks])
        fa = np.array([idx_on[k]["foot"] for k in ks])
        fb = np.array([idx_off[k]["foot"] for k in ks])
        pm = mcnemar(int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum()))
        pf = mcnemar(int(((fa == 1) & (fb == 0)).sum()), int(((fa == 0) & (fb == 1)).sum()))
        dc = float(np.mean([idx_on[k]["crest"] - idx_off[k]["crest"] for k in ks]))
        da = float(np.median([idx_on[k]["attack"] - idx_off[k]["attack"] for k in ks]))
        e = {"factor": fname, "pairs": len(ks), "mat_on": float(a.mean()), "mat_off": float(b.mean()),
             "mat_p": pm, "foot_on": float(fa.mean()), "foot_off": float(fb.mean()), "foot_p": pf,
             "d_crest": dc, "d_attack_median": da}
        effects.append(e)
        print("  %-10s %6d %6.0f%%-%3.0f%% %9.4f %6.0f%%-%3.0f%%   %9.4f %+8.2f %+8.2f" % (
            fname, len(ks), 100 * e["mat_on"], 100 * e["mat_off"], pm, 100 * e["foot_on"], 100 * e["foot_off"],
            pf, dc, da))
    print("=" * W)
    json.dump({"args": vars(args), "cells": cells, "effects": effects, "rows": rows}, open(args.out, "w"), indent=1)
    print("saved -> %s" % args.out)


if __name__ == "__main__":
    main()

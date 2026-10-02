import _bootstrap

import argparse
import gc
import json
import os
import time
from math import comb, sqrt

import numpy as np
import torch

from sarge.generators.stable_audio import _MATERIAL_ADJ, _PROMPT_TEMPLATES, _extract, _seed, build_prompt
from sarge.generators.stable_audio3 import StableAudio3Generator
from sarge.loudness import normalize_loudness, target_for
from sarge.types import AudioClip, CacheKey
from validate_panel import CATEGORIES, _resample
from gate_quality import transients

CLAP_NAME = "laion/clap-htsat-unfused"
FOOT_IDX = CATEGORIES.index("a footstep")
W = 142


def prompt_for(kind, mat):
    if kind == "plain":
        return _PROMPT_TEMPLATES["footstep"].format(intensity="normal", material=mat, event="footstep")
    if kind == "enriched":
        return build_prompt("footstep", mat, "normal")
    if kind == "adjonly":
        return "a %s footstep on %s, dry, isolated sound effect" % (_MATERIAL_ADJ.get(mat, mat), mat)
    raise ValueError(kind)


def gen_one(g, prompt, steps, sampler, seed):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        audio = g._model.generate(prompt=prompt, duration=g.gen_duration_s, steps=steps,
                                  cfg_scale=g.cfg_scale, seed=seed, sampler_type=sampler,
                                  disable_tqdm=True)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000.0
    a = audio.detach().to(torch.float32).cpu().numpy()
    while a.ndim > 2:
        a = a[0]
    mono = a.mean(axis=0) if a.ndim == 2 else a
    sr = int(g.sample_rate)
    tgt = target_for("normal")
    x06 = normalize_loudness(_extract(mono, sr, g.duration_s), target_rms=tgt)
    x10 = normalize_loudness(mono, target_rms=tgt)
    return x06.astype(np.float32), x10.astype(np.float32), ms


def clap_scores(sets, text_sets, dev, bs):
    from transformers import ClapModel, ClapProcessor
    m = ClapModel.from_pretrained(CLAP_NAME).to(dev).eval()
    pr = ClapProcessor.from_pretrained(CLAP_NAME)
    toks = {k: pr(text=t, return_tensors="pt", padding=True) for k, t in text_sets.items()}
    out = {}
    for sname, clips in sets.items():
        res = {k: [] for k in text_sets}
        for i in range(0, len(clips), bs):
            a = pr(audio=clips[i: i + bs], sampling_rate=48000, return_tensors="pt")
            for k, t in toks.items():
                inp = {kk: vv.to(dev) for kk, vv in {**a, **t}.items()}
                with torch.no_grad():
                    res[k].append(m(**inp).logits_per_audio.float().cpu().numpy())
        out[sname] = {k: np.concatenate(v, 0) for k, v in res.items()}
    del m
    torch.cuda.empty_cache()
    return out


def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="plain,enriched")
    ap.add_argument("--samplers", default="euler,pingpong")
    ap.add_argument("--steps", default="3,4,8")
    ap.add_argument("--variants", type=int, default=10)
    ap.add_argument("--materials", nargs="+", default=["wood", "stone", "grass", "metal"])
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--out", default=os.path.join("data", "_gate", "ab_results.json"))
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    prompts = args.prompts.split(",")
    samplers = args.samplers.split(",")
    steps_list = [int(s) for s in args.steps.split(",")]
    mats = args.materials

    g = StableAudio3Generator()
    g.ensure_ready()
    sr = int(g.sample_rate)
    print("model small-sfx fp16=%s cfg=%.1f gen=%.1fs extract=%.1fs sr=%d"
          % (g.model_half, g.cfg_scale, g.gen_duration_s, g.duration_s, sr))
    for s in samplers:
        gen_one(g, prompt_for(prompts[0], mats[0]), 2, s, 0)

    cells, rows, c06, c10 = [], [], [], []
    for p in prompts:
        for s in samplers:
            for st in steps_list:
                tag = "ab_%s_%s_s%d" % (p, s, st)
                cells.append(tag)
                d = os.path.join("data", "_gate", tag)
                os.makedirs(d, exist_ok=True)
                ms = []
                for mat in mats:
                    key = CacheKey("footstep", mat, "normal")
                    text = prompt_for(p, mat)
                    for v in range(args.variants):
                        x06, x10, t = gen_one(g, text, st, s, _seed(key, v))
                        AudioClip(x06, sr).to_wav(os.path.join(d, "%s_%02d.wav" % (mat, v)))
                        AudioClip(x10, sr).to_wav(os.path.join(d, "%s_%02d_full.wav" % (mat, v)))
                        rows.append({"cell": tag, "prompt": p, "sampler": s, "steps": st,
                                     "mat": mat, "v": v, "ms": t, "text": text})
                        c06.append(x06)
                        c10.append(x10)
                        ms.append(t)
                print("generated %-28s %3d clips  median %.1f ms" % (tag, len(ms), np.median(ms)))
    for p in prompts:
        print("prompt %-9s %s" % (p, prompt_for(p, mats[0])))

    g._model = None
    del g
    gc.collect()
    torch.cuda.empty_cache()

    text_sets = {"mat": ["the sound of a footstep on %s" % x for x in mats],
                 "alt": ["walking on %s" % x for x in mats],
                 "cat": CATEGORIES}
    sets = {"x06": [_resample(x, sr, 48000) for x in c06],
            "x10": [_resample(x, sr, 48000) for x in c10]}
    sc = clap_scores(sets, text_sets, dev, args.bs)

    true = np.array([mats.index(r["mat"]) for r in rows])
    for ln in ("x06", "x10"):
        ml = sc[ln]["mat"]
        pt = softmax(ml)[np.arange(len(rows)), true]
        pred = ml.argmax(1)
        alt = sc[ln]["alt"].argmax(1)
        foot = sc[ln]["cat"].argmax(1) == FOOT_IDX
        for i, r in enumerate(rows):
            r["hit_" + ln] = int(pred[i] == true[i])
            r["pt_" + ln] = float(pt[i])
            r["pred_" + ln] = mats[int(pred[i])]
            r["alt_" + ln] = int(alt[i] == true[i])
            r["foot_" + ln] = int(foot[i])

    def sel(**kw):
        return [r for r in rows if all(r[k] == v for k, v in kw.items())]

    def frac(rs, f):
        return "%d/%d" % (sum(r[f] for r in rs), len(rs))

    summ = {}
    print()
    print("=" * W)
    print(" CLAP MATERIAL A/B   %d materials x %d variants per cell, chance %.0f%%, judge text 'the sound of a footstep on X'"
          % (len(mats), args.variants, 100.0 / len(mats)))
    print("=" * W)
    hdr = "%-28s %6s %7s %7s %6s %6s %5s %5s %5s %6s %6s %6s  %-12s %-12s"
    print(hdr % ("cell", "ms", "acc06", "acc10", "pT06", "pT10", "ft06", "ft10", "alt06",
                 "rep5", "crest", "atk", "hit " + "/".join(m[0] for m in mats),
                 "pred " + "/".join(m[0] for m in mats)))
    print("-" * W)
    for tag in cells:
        rs = sel(cell=tag)
        n = len(rs)
        tr = transients([c06[i] for i, r in enumerate(rows) if r["cell"] == tag], sr)
        k06 = sum(r["hit_x06"] for r in rs)
        k10 = sum(r["hit_x10"] for r in rs)
        r5 = [r for r in rs if r["v"] < 5]
        per = "/".join(str(sum(r["hit_x06"] for r in rs if r["mat"] == m)) for m in mats)
        prd = "/".join(str(sum(1 for r in rs if r["pred_x06"] == m)) for m in mats)
        summ[tag] = {"n": n, "gen_ms": float(np.median([r["ms"] for r in rs])),
                     "acc06": k06 / n, "acc10": k10 / n,
                     "acc06_ci": wilson(k06, n), "acc10_ci": wilson(k10, n),
                     "pt06": float(np.mean([r["pt_x06"] for r in rs])),
                     "pt10": float(np.mean([r["pt_x10"] for r in rs])),
                     "foot06": float(np.mean([r["foot_x06"] for r in rs])),
                     "foot10": float(np.mean([r["foot_x10"] for r in rs])),
                     "alt06": float(np.mean([r["alt_x06"] for r in rs])),
                     "rep5_06": frac(r5, "hit_x06"),
                     "rep4_06": frac([r for r in rs if r["v"] < 4], "hit_x06"),
                     "hit_per_mat06": per, "pred_hist06": prd, "transients06": tr}
        s = summ[tag]
        print("%-28s %6.1f %7s %7s %6.3f %6.3f %4.0f%% %4.0f%% %4.0f%% %6s %6.1f %6.2f  %-12s %-12s" % (
            tag, s["gen_ms"], "%d/%d" % (k06, n), "%d/%d" % (k10, n), s["pt06"], s["pt10"],
            100 * s["foot06"], 100 * s["foot10"], 100 * s["alt06"], s["rep5_06"],
            tr["crest_db"], tr["attack_ms"], per, prd))
    print("-" * W)
    print(" acc = argmax over the %d judge texts; pT = mean softmax prob on the true material (chance %.3f);"
          % (len(mats), 1.0 / len(mats)))
    print(" x06 = 0.6 s onset extract (what ships and what every past run scored), x10 = full 1 s clip;")
    print(" ft = CLAP picks 'a footstep' among %d categories; alt = judge text 'walking on X'; rep5 = variants 0-4 only"
          % len(CATEGORIES))
    print(" crest/atk = gate_quality.transients on x06 (real footsteps: crest 18.7 dB, attack 10.40 ms)")

    print()
    print(" ANCHORS (same seeds as the historical runs; rep5 = variants 0-4)")
    anchors = [("ab_plain_pingpong_s4", "rep5_06", "June 26 bench_quality 4 steps: 14/20"),
               ("ab_plain_pingpong_s8", "rep5_06", "June 26 bench_quality 8 steps: 20/20"),
               ("ab_plain_pingpong_s8", "rep4_06", "June 22 optimize.py sa3_cfg2 8 steps: 16/16 (peak-norm, not RMS-norm)"),
               ("ab_enriched_euler_s4", "rep5_06", "Sep 27 gate steps4: 5/20"),
               ("ab_enriched_euler_s3", "rep5_06", "Sep 27 gate steps3: 10/20")]
    for tag, f, ref in anchors:
        if tag in summ:
            print("  %-28s %-8s %6s   vs %s" % (tag, f[:4], summ[tag][f], ref))
    for st in (3, 4):
        tag = "ab_enriched_euler_s%d" % st
        ref = os.path.join("data", "_gate", "steps%d" % st)
        if tag not in summ or not os.path.isdir(ref):
            continue
        diffs = []
        for mat in mats:
            for v in range(min(5, args.variants)):
                pa = os.path.join(ref, "%s_%02d.wav" % (mat, v))
                if not os.path.exists(pa):
                    continue
                a = AudioClip.from_wav(pa).samples
                b = AudioClip.from_wav(os.path.join("data", "_gate", tag, "%s_%02d.wav" % (mat, v))).samples
                n = min(len(a), len(b))
                diffs.append(float(np.max(np.abs(a[:n] - b[:n]))))
        if diffs:
            print("  determinism %s vs %s: %d clips, max|diff| %.5f, identical %d"
                  % (tag, ref, len(diffs), max(diffs), sum(d == 0.0 for d in diffs)))

    print()
    print(" MAIN EFFECTS (pooled over all other factors)")
    print("  %-22s %9s %15s %7s" % ("level", "acc", "wilson95", "pT"))
    levels = [("prompt", p) for p in prompts] + [("sampler", s) for s in samplers] + [("steps", st) for st in steps_list]
    for ln in ("x06", "x10"):
        for f, lv in levels:
            rs = [r for r in rows if r[f] == lv]
            k, n = sum(r["hit_" + ln] for r in rs), len(rs)
            lo, hi = wilson(k, n)
            print("  %-22s %4d/%-4d [%.2f, %.2f] %7.3f" % ("%s %s=%s" % (ln, f, lv), k, n, lo, hi,
                                                         np.mean([r["pt_" + ln] for r in rs])))

    print()
    print(" PAIRED CONTRASTS on x06 (same seed/material/other factors; b = A right B wrong, c = A wrong B right)")
    print("  %-30s %5s %6s %6s %4s %4s %9s %9s" % ("A vs B", "pairs", "accA", "accB", "b", "c", "mcnemarP", "dPT"))
    contrasts = []
    for f, lv in (("prompt", prompts), ("sampler", samplers), ("steps", sorted(steps_list, reverse=True))):
        for i in range(len(lv)):
            for j in range(i + 1, len(lv)):
                contrasts.append((f, lv[i], lv[j]))
    keyf = ("prompt", "sampler", "steps", "mat", "v")
    res_c = []
    for f, a, b in contrasts:
        other = [k for k in keyf if k != f]
        ia = {tuple(r[k] for k in other): r for r in rows if r[f] == a}
        ib = {tuple(r[k] for k in other): r for r in rows if r[f] == b}
        ks = [k for k in ia if k in ib]
        if not ks:
            continue
        ha = np.array([ia[k]["hit_x06"] for k in ks])
        hb = np.array([ib[k]["hit_x06"] for k in ks])
        dp = np.array([ia[k]["pt_x06"] - ib[k]["pt_x06"] for k in ks])
        bb, cc = int(((ha == 1) & (hb == 0)).sum()), int(((ha == 0) & (hb == 1)).sum())
        pv = mcnemar(bb, cc)
        se = float(dp.std(ddof=1) / sqrt(len(dp))) if len(dp) > 1 else 0.0
        res_c.append({"factor": f, "a": a, "b": b, "pairs": len(ks), "acc_a": float(ha.mean()),
                      "acc_b": float(hb.mean()), "b_only_a": bb, "c_only_b": cc, "p": pv,
                      "dpt": float(dp.mean()), "dpt_se": se})
        print("  %-30s %5d %6.2f %6.2f %4d %4d %9.4f %+.3f+-%.3f" % (
            "%s %s vs %s" % (f, a, b), len(ks), ha.mean(), hb.mean(), bb, cc, pv, dp.mean(), se))
    h06 = np.array([r["hit_x06"] for r in rows])
    h10 = np.array([r["hit_x10"] for r in rows])
    dl = np.array([r["pt_x10"] - r["pt_x06"] for r in rows])
    bb, cc = int(((h10 == 1) & (h06 == 0)).sum()), int(((h10 == 0) & (h06 == 1)).sum())
    pv = mcnemar(bb, cc)
    res_c.append({"factor": "length", "a": "x10", "b": "x06", "pairs": len(rows), "acc_a": float(h10.mean()),
                  "acc_b": float(h06.mean()), "b_only_a": bb, "c_only_b": cc, "p": pv,
                  "dpt": float(dl.mean()), "dpt_se": float(dl.std(ddof=1) / sqrt(len(dl)))})
    print("  %-30s %5d %6.2f %6.2f %4d %4d %9.4f %+.3f+-%.3f" % (
        "length x10 vs x06", len(rows), h10.mean(), h06.mean(), bb, cc, pv, dl.mean(),
        dl.std(ddof=1) / sqrt(len(dl))))
    print("=" * W)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"args": vars(args), "cells": summ, "contrasts": res_c, "rows": rows},
              open(args.out, "w"), indent=1)
    print("saved -> %s" % args.out)


if __name__ == "__main__":
    main()

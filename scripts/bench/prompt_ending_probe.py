import _bootstrap

import argparse
import gc
import json
import os

import numpy as np
import torch
from scipy.io import wavfile

from sarge.generators.stable_audio import build_prompt
from sarge.generators.stable_audio3 import StableAudio3Generator
from sarge.types import CacheKey
from validate_panel import CATEGORIES, _resample
from gate_quality import crest_db, attack_ms, onset_window
from ab_clap import clap_scores, mcnemar
from score_hard import HARD

MATS = ["wood", "stone", "grass", "metal"]
SHIPPED_END = ", dry, isolated sound effect"
ENDINGS = {"dry": SHIPPED_END, "none": "", "room": ", recorded in a quiet room"}
PREFIXES = {"plain": "", "tt": "TrackType: SFX, "}


def prompt_for(mat, prefix, ending):
    base = build_prompt("footstep", mat, "normal")
    assert base.endswith(SHIPPED_END)
    return PREFIXES[prefix] + base[: -len(SHIPPED_END)] + ENDINGS[ending]


def gap_features(x, sr):
    a = np.asarray(x, dtype=np.float64)
    r = np.sqrt(np.mean(a * a)) + 1e-12
    w = int(0.01 * sr)
    n = len(a) // w
    fr = np.sqrt(np.mean(a[: n * w].reshape(n, w) ** 2, axis=1)) + 1e-12
    edge = int(0.05 * sr)
    inner = a[edge: len(a) - edge]
    pk = int(np.argmax(fr))
    body = fr[pk + 5: pk + 30]
    return {"floor_db": float(20 * np.log10(np.percentile(fr, 10) / r)),
            "zeros": float(np.mean(inner == 0.0)) if inner.size else 0.0,
            "body_db": float(20 * np.log10(np.sqrt(np.mean(body ** 2)) / fr[pk])) if body.size else -120.0,
            "hold20_ms": float(10.0 * np.sum(fr[pk:] >= fr[pk] * 0.1))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--out", default=os.path.join("scripts", "bench", "out_prompt_ending.json"))
    args = ap.parse_args()

    g = StableAudio3Generator()
    g.ensure_ready()
    sr = int(g.sample_rate)
    cells = [(p, e) for p in PREFIXES for e in ENDINGS]
    rows, clips = [], []
    for p, e in cells:
        tag = "%s_%s" % (p, e)
        d = os.path.join("data", "_gate", "ending", tag)
        os.makedirs(d, exist_ok=True)
        for mat in MATS:
            key = CacheKey("footstep", mat, "normal")
            g.prompts[str(key)] = prompt_for(mat, p, e)
            for v in range(args.seeds):
                c = g.generate(key, variant=v)
                path = os.path.join(d, "%s_%02d.wav" % (mat, v))
                c.to_wav(path)
                w = onset_window(c.samples, sr)
                r = {"cell": tag, "mat": mat, "v": v, "path": path, "crest": crest_db(w), "attack": attack_ms(w, sr)}
                r.update(gap_features(c.samples, sr))
                rows.append(r)
                clips.append(_resample(c.samples, sr, 48000))
            g.prompts.pop(str(key), None)
        print("generated %-12s %s" % (tag, prompt_for("metal", p, e)))
    stats = g.fast_stats()
    del g
    gc.collect()
    torch.cuda.empty_cache()

    sc = clap_scores({"all": clips}, {"base": CATEGORIES, "hard": HARD,
                                      "mat": ["the sound of a footstep on %s" % m for m in MATS]}, "cuda", 32)["all"]
    for i, r in enumerate(rows):
        r["foot"] = int(sc["base"][i].argmax() == CATEGORIES.index("a footstep"))
        r["hard"] = int(sc["hard"][i].argmax() == HARD.index("a footstep"))
        r["matok"] = int(MATS[int(sc["mat"][i].argmax())] == r["mat"])

    from audiobox_aesthetics.infer import initialize_predictor
    pred = initialize_predictor()
    for i in range(0, len(rows), 40):
        chunk = rows[i: i + 40]
        for r, o in zip(chunk, pred.forward([{"path": r["path"]} for r in chunk])):
            r["PQ"] = float(o["PQ"])

    ref = {(r["mat"], r["v"]): r for r in rows if r["cell"] == "plain_dry"}
    out = {"cells": {}, "fast_stats": stats}
    print()
    print(" PROMPT PREFIX x ENDING  (footsteps, %d paired seeds per cell, vs shipped plain_dry)" % (len(MATS) * args.seeds))
    print(" real ESC-50 footstep crops for reference: hard CLAP 87.5%%, floor about -17 dB re RMS, zeros 0%%, longer body")
    print("%-12s %17s %17s %17s %7s %6s %7s %8s %8s %6s" % ("cell", "clap footstep", "clap hard", "material", "floor",
                                                         "zeros", "body", "hold20", "attack", "PQ"))
    for p, e in cells:
        tag = "%s_%s" % (p, e)
        rs = {(r["mat"], r["v"]): r for r in rows if r["cell"] == tag}
        ks = [k for k in ref if k in rs]
        res = {}
        cols = []
        for f in ("foot", "hard", "matok"):
            a = np.array([rs[k][f] for k in ks])
            b = np.array([ref[k][f] for k in ks])
            gn, ls = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
            res[f] = {"rate": float(a.mean()), "gain": gn, "loss": ls, "p": mcnemar(gn, ls)}
            cols.append("%3.0f%% +%d/-%d p%.2f" % (100 * a.mean(), gn, ls, mcnemar(gn, ls)))
        for f in ("floor_db", "zeros", "body_db", "hold20_ms", "attack", "PQ", "crest"):
            res[f] = float(np.median([rs[k][f] for k in ks]))
        res["per_mat_hard"] = {m: float(np.mean([rs[k]["hard"] for k in ks if k[0] == m])) for m in MATS}
        out["cells"][tag] = res
        print("%-12s %17s %17s %17s %7.1f %5.1f%% %7.1f %8.0f %8.2f %6.2f" % (
            tag, cols[0], cols[1], cols[2], res["floor_db"], 100 * res["zeros"], res["body_db"], res["hold20_ms"],
            res["attack"], res["PQ"]))
    print(" floor = 10th percentile 10 ms frame level re clip RMS; zeros = exact zero samples inside the clip;")
    print(" body = level 50 to 300 ms after the peak re the peak frame; hold20 = ms within 20 dB of the peak")

    seq = []
    for p, e in cells:
        for mat in ("wood", "grass"):
            for v in (0, 1):
                s_, a = wavfile.read(os.path.join("data", "_gate", "ending", "%s_%s" % (p, e), "%s_%02d.wav" % (mat, v)))
                a = a.astype(np.float32)
                if np.abs(a).max() > 1.5:
                    a = a / 32768.0
                seq += [a, np.zeros(int(0.3 * sr), np.float32)]
        seq.append(np.zeros(int(1.0 * sr), np.float32))
    x = np.concatenate(seq)
    x = x / max(1e-9, np.abs(x).max()) * 0.9
    lp = os.path.join("data", "_gate", "listen", "prompt_endings.wav")
    wavfile.write(lp, sr, (x * 32767).astype(np.int16))
    print("listening file %s: per cell wood v0, wood v1, grass v0, grass v1; cells in order %s" % (
        lp, " | ".join("%s_%s" % c for c in cells)))
    json.dump({"result": out, "rows": rows}, open(args.out, "w"), indent=1)
    print("saved -> %s" % args.out)


if __name__ == "__main__":
    main()

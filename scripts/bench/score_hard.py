import _bootstrap

import argparse
import glob
import json
import os
import re

import numpy as np
import torch

from sarge.types import AudioClip
from validate_panel import CATEGORIES, _resample
from gate_quality import crest_db, attack_ms, onset_window
from ab_clap import clap_scores, mcnemar

DISTRACTORS = ["hands clapping", "typing on a keyboard", "a hammer hitting", "a ball bouncing",
               "a clock ticking", "a door closing"]
HARD = CATEGORIES + DISTRACTORS
FACTORS = ["single", "normal", "adj", "clear", "closemic"]
NAME = re.compile(r"^([a-z]+)_(\d\d)\.wav$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="pf_")
    ap.add_argument("--root", default=os.path.join("data", "_gate"))
    ap.add_argument("--out", default=os.path.join("scripts", "bench", "out_score_hard.json"))
    args = ap.parse_args()

    rows, clips = [], []
    for d in sorted(glob.glob(os.path.join(args.root, args.prefix + "*"))):
        code = os.path.basename(d)[len(args.prefix):]
        for p in sorted(glob.glob(os.path.join(d, "*.wav"))):
            m = NAME.match(os.path.basename(p))
            if not m:
                continue
            c = AudioClip.from_wav(p)
            w = onset_window(c.samples, c.sample_rate)
            clips.append(_resample(c.samples, c.sample_rate, 48000))
            rows.append({"code": code, "mat": m.group(1), "v": int(m.group(2)),
                         "crest": crest_db(w), "attack": attack_ms(w, c.sample_rate)})
    sc = clap_scores({"all": clips}, {"hard": HARD}, "cuda" if torch.cuda.is_available() else "cpu", 32)["all"]["hard"]
    top = sc.argmax(1)
    for i, r in enumerate(rows):
        r["hard_foot"] = int(top[i] == HARD.index("a footstep"))
        r["hard_top1"] = HARD[int(top[i])]

    codes = sorted({r["code"] for r in rows})
    cells = []
    for c in codes:
        rs = [r for r in rows if r["code"] == c]
        tops = {}
        for r in rs:
            tops[r["hard_top1"]] = tops.get(r["hard_top1"], 0) + 1
        cells.append({"code": c, "n": len(rs), "hard_foot": float(np.mean([r["hard_foot"] for r in rs])),
                      "attack": float(np.median([r["attack"] for r in rs])),
                      "crest": float(np.mean([r["crest"] for r in rs])),
                      "top": sorted(tops.items(), key=lambda kv: -kv[1])[:3]})
    cells.sort(key=lambda c: -c["hard_foot"])
    print()
    print(" HARD CLAP (13-way incl. 6 percussive distractors)  real matched crops 87.5%, attack 12.78 ms")
    print("%-8s %4s %8s %8s %7s  %s" % ("code", "n", "hardFt", "attack", "crest", "top1 counts"))
    for c in cells:
        print("%-8s %4d %7.0f%% %8.2f %7.1f  %s" % (c["code"], c["n"], 100 * c["hard_foot"], c["attack"], c["crest"], c["top"]))

    effects = []
    if all(len(c) == len(FACTORS) and set(c) <= {"0", "1"} for c in codes):
        print()
        print(" MAIN EFFECTS on hard footstep (paired over the other factors, material, seed)")
        for fi, fname in enumerate(FACTORS):
            on = {(r["code"][:fi] + r["code"][fi + 1:], r["mat"], r["v"]): r for r in rows if r["code"][fi] == "1"}
            off = {(r["code"][:fi] + r["code"][fi + 1:], r["mat"], r["v"]): r for r in rows if r["code"][fi] == "0"}
            ks = [k for k in on if k in off]
            if not ks:
                continue
            a = np.array([on[k]["hard_foot"] for k in ks])
            b = np.array([off[k]["hard_foot"] for k in ks])
            g, l = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
            da = float(np.median([on[k]["attack"] - off[k]["attack"] for k in ks]))
            e = {"factor": fname, "pairs": len(ks), "on": float(a.mean()), "off": float(b.mean()),
                 "gain": g, "loss": l, "p": mcnemar(g, l), "d_attack_median": da}
            effects.append(e)
            print("  %-9s %5d pairs  %4.0f%% vs %4.0f%%  +%d/-%d p=%.4f  attack %+.2f ms" % (
                fname, len(ks), 100 * e["on"], 100 * e["off"], g, l, e["p"], da))
    json.dump({"cells": cells, "effects": effects, "rows": rows}, open(args.out, "w"), indent=1)
    print("saved -> %s" % args.out)


if __name__ == "__main__":
    main()

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

FOOT_IDX = CATEGORIES.index("a footstep")
NAME = re.compile(r"^([a-z]+)_(\d\d)\.wav$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--materials", nargs="+", default=["wood", "stone", "grass", "metal"])
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    mats = args.materials

    sets = {}
    for d in sorted(glob.glob(os.path.join(args.root, "*"))):
        if not os.path.isdir(d):
            continue
        items = []
        for p in sorted(glob.glob(os.path.join(d, "*.wav"))):
            m = NAME.match(os.path.basename(p))
            if m and m.group(1) in mats:
                items.append((m.group(1), int(m.group(2)), p))
        if items:
            sets[os.path.basename(d)] = items
    if args.ref not in sets:
        raise SystemExit("ref %s not in %s" % (args.ref, sorted(sets)))

    clips, rows = [], []
    for name, items in sets.items():
        for mat, v, p in items:
            c = AudioClip.from_wav(p)
            clips.append(_resample(c.samples, c.sample_rate, 48000))
            w = onset_window(c.samples, c.sample_rate)
            rows.append({"cfg": name, "mat": mat, "v": v, "crest": crest_db(w), "attack": attack_ms(w, c.sample_rate)})
    sc = clap_scores({"all": clips}, {"mat": ["the sound of a footstep on %s" % x for x in mats], "cat": CATEGORIES},
                     "cuda" if torch.cuda.is_available() else "cpu", 32)["all"]
    pred = sc["mat"].argmax(1)
    foot = sc["cat"].argmax(1) == FOOT_IDX
    for i, r in enumerate(rows):
        r["hit"] = int(mats[int(pred[i])] == r["mat"])
        r["foot"] = int(foot[i])

    ref = {(r["mat"], r["v"]): r for r in rows if r["cfg"] == args.ref}
    out = []
    print()
    print("%-24s %4s %6s %6s %7s %7s   %-26s %-26s" % ("config", "n", "mat", "foot", "crest", "attack",
                                                    "mat vs ref (+/-, p)", "foot vs ref (+/-, p)"))
    print("-" * 116)
    for name in sets:
        rs = {(r["mat"], r["v"]): r for r in rows if r["cfg"] == name}
        ks = [k for k in rs if k in ref]
        res = {"cfg": name, "n": len(rs), "mat": float(np.mean([r["hit"] for r in rs.values()])),
               "foot": float(np.mean([r["foot"] for r in rs.values()])),
               "crest": float(np.mean([r["crest"] for r in rs.values()])),
               "attack": float(np.median([r["attack"] for r in rs.values()]))}
        cols = []
        for f in ("hit", "foot"):
            a = np.array([rs[k][f] for k in ks])
            b = np.array([ref[k][f] for k in ks])
            g, l = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
            p = mcnemar(g, l)
            res[f + "_gain"], res[f + "_loss"], res[f + "_p"] = g, l, p
            cols.append("+%d/-%d p=%.3f" % (g, l, p))
        out.append(res)
        print("%-24s %4d %5.0f%% %5.0f%% %7.1f %7.2f   %-26s %-26s" % (
            name, res["n"], 100 * res["mat"], 100 * res["foot"], res["crest"], res["attack"], cols[0], cols[1]))
    if args.out:
        json.dump({"ref": args.ref, "configs": out, "rows": rows}, open(args.out, "w"), indent=1)
        print("saved -> %s" % args.out)


if __name__ == "__main__":
    main()

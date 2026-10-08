import _bootstrap

import argparse
import gc
import json
import os

import numpy as np
import torch
from scipy.io import wavfile

from sarge.generators.stable_audio import _MATERIAL_ADJ, _FOOTSTEP_PLAIN
from sarge.generators.stable_audio3 import StableAudio3Generator
from sarge.types import CacheKey
from validate_panel import CATEGORIES, _resample
from gate_quality import crest_db, attack_ms, onset_window
from ab_clap import clap_scores, mcnemar
from score_hard import HARD
from prompt_ending_probe import gap_features

MATS = ["wood", "stone", "grass", "metal"]


def adj(m):
    return "" if m in _FOOTSTEP_PLAIN else _MATERIAL_ADJ.get(m, m) + " "


CELLS = {
    "shipped": lambda m: "a normal %sfootstep on %s, a clear distinct footstep, dry, isolated sound effect" % (adj(m), m),
    "room": lambda m: "a normal %sfootstep on %s, a clear distinct footstep, recorded in a quiet room" % (adj(m), m),
    "room_floor": lambda m: "a normal %sfootstep on a %s floor, a clear distinct footstep, recorded in a quiet room" % (adj(m), m),
    "room_then_mat": lambda m: "a normal %sfootstep on %s, a clear distinct footstep, recorded in a quiet room, %s surface" % (adj(m), m, m),
    "room_closemic": lambda m: "a normal %sfootstep on %s, a clear distinct footstep, close mic recording in a quiet room" % (adj(m), m),
    "dry_and_room": lambda m: "a normal %sfootstep on %s, a clear distinct footstep, dry, isolated sound effect, recorded in a quiet room" % (adj(m), m),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--v0", type=int, default=0)
    ap.add_argument("--cells", default=",".join(CELLS))
    ap.add_argument("--tag", default="room")
    ap.add_argument("--out", default=os.path.join("scripts", "bench", "out_prompt_room.json"))
    args = ap.parse_args()
    names = args.cells.split(",")

    g = StableAudio3Generator()
    g.ensure_ready()
    sr = int(g.sample_rate)
    rows, clips = [], []
    for name in names:
        d = os.path.join("data", "_gate", args.tag, name)
        os.makedirs(d, exist_ok=True)
        for mat in MATS:
            key = CacheKey("footstep", mat, "normal")
            g.prompts[str(key)] = CELLS[name](mat)
            for v in range(args.v0, args.v0 + args.seeds):
                c = g.generate(key, variant=v)
                path = os.path.join(d, "%s_%02d.wav" % (mat, v))
                c.to_wav(path)
                w = onset_window(c.samples, sr)
                r = {"cell": name, "mat": mat, "v": v, "path": path, "crest": crest_db(w), "attack": attack_ms(w, sr)}
                r.update(gap_features(c.samples, sr))
                rows.append(r)
                clips.append(_resample(c.samples, sr, 48000))
            g.prompts.pop(str(key), None)
        print("generated %-14s %s" % (name, CELLS[name]("metal")))
    del g
    gc.collect()
    torch.cuda.empty_cache()

    sc = clap_scores({"all": clips}, {"base": CATEGORIES, "hard": HARD,
                                      "mat": ["the sound of a footstep on %s" % m for m in MATS]}, "cuda", 32)["all"]
    for i, r in enumerate(rows):
        r["foot"] = int(sc["base"][i].argmax() == CATEGORIES.index("a footstep"))
        r["hard"] = int(sc["hard"][i].argmax() == HARD.index("a footstep"))
        r["matok"] = int(MATS[int(sc["mat"][i].argmax())] == r["mat"])
        r["both"] = int(r["hard"] and r["matok"])

    from audiobox_aesthetics.infer import initialize_predictor
    pred = initialize_predictor()
    for i in range(0, len(rows), 40):
        chunk = rows[i: i + 40]
        for r, o in zip(chunk, pred.forward([{"path": r["path"]} for r in chunk])):
            r["PQ"] = float(o["PQ"])

    ref_name = names[0]
    ref = {(r["mat"], r["v"]): r for r in rows if r["cell"] == ref_name}
    out = {}
    print()
    print(" ROOM WORDINGS (footsteps, %d paired seeds per cell from variant %d, vs %s)" % (len(MATS) * args.seeds, args.v0, ref_name))
    print("%-14s %17s %17s %17s %17s %7s %8s %6s  %s" % ("cell", "clap footstep", "clap hard", "material", "hard+material",
                                                      "body", "attack", "PQ", "hard by material w/s/g/m"))
    for name in names:
        rs = {(r["mat"], r["v"]): r for r in rows if r["cell"] == name}
        ks = [k for k in ref if k in rs]
        res, cols = {}, []
        for f in ("foot", "hard", "matok", "both"):
            a = np.array([rs[k][f] for k in ks])
            b = np.array([ref[k][f] for k in ks])
            gn, ls = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
            res[f] = {"rate": float(a.mean()), "gain": gn, "loss": ls, "p": mcnemar(gn, ls)}
            cols.append("%3.0f%% +%d/-%d p%.2f" % (100 * a.mean(), gn, ls, mcnemar(gn, ls)))
        for f in ("body_db", "attack", "PQ", "floor_db", "hold20_ms"):
            res[f] = float(np.median([rs[k][f] for k in ks]))
        per = "/".join("%.0f" % (100 * np.mean([rs[k]["hard"] for k in ks if k[0] == m])) for m in MATS)
        res["prompt_metal"] = CELLS[name]("metal")
        out[name] = res
        print("%-14s %17s %17s %17s %17s %7.1f %8.2f %6.2f  %s" % (name, cols[0], cols[1], cols[2], cols[3],
                                                               res["body_db"], res["attack"], res["PQ"], per))

    seq = []
    for name in names:
        for mat in ("wood", "metal", "grass"):
            s_, a = wavfile.read(os.path.join("data", "_gate", args.tag, name, "%s_%02d.wav" % (mat, args.v0)))
            a = a.astype(np.float32)
            if np.abs(a).max() > 1.5:
                a = a / 32768.0
            seq += [a, np.zeros(int(0.3 * sr), np.float32)]
        seq.append(np.zeros(int(1.0 * sr), np.float32))
    x = np.concatenate(seq)
    x = x / max(1e-9, np.abs(x).max()) * 0.9
    lp = os.path.join("data", "_gate", "listen", "%s_wordings.wav" % args.tag)
    wavfile.write(lp, sr, (x * 32767).astype(np.int16))
    print("listening file %s: per cell wood, metal, grass; cells in order %s" % (lp, " | ".join(names)))
    json.dump({"result": out, "rows": rows}, open(args.out, "w"), indent=1)
    print("saved -> %s" % args.out)


if __name__ == "__main__":
    main()

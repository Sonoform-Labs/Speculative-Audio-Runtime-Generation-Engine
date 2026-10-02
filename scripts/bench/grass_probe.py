import _bootstrap

import gc
import json
import os

import numpy as np
import torch
from scipy.io import wavfile

from sarge.generators.stable_audio3 import StableAudio3Generator
from sarge.types import CacheKey
from validate_panel import CATEGORIES, _resample
from gate_quality import crest_db, attack_ms, onset_window
from ab_clap import clap_scores, mcnemar
from score_hard import HARD

TAIL = ", a clear distinct footstep, dry, isolated sound effect"
CANDIDATES = {
    "grassy_on_grass": "a normal grassy footstep on grass" + TAIL,
    "on_grass": "a normal footstep on grass" + TAIL,
    "on_short_grass": "a normal footstep on short grass" + TAIL,
    "soft_on_short_grass": "a normal soft footstep on short grass" + TAIL,
    "on_a_lawn": "a normal footstep on a lawn" + TAIL,
    "muffled_on_trimmed_grass": "a normal muffled footstep on short trimmed grass" + TAIL,
}
MATS = ["wood", "stone", "grass", "metal"]
RUSTLE = ["walking through bushes", "rustling leaves", "a footstep on short grass", "a footstep on dirt"]
VARIANTS = 20


def main():
    g = StableAudio3Generator()
    g.ensure_ready()
    sr = int(g.sample_rate)
    key = CacheKey("footstep", "grass", "normal")
    rows, clips = [], []
    for name, text in CANDIDATES.items():
        g.prompts[str(key)] = text
        d = os.path.join("data", "_gate", "grass", name)
        os.makedirs(d, exist_ok=True)
        for v in range(VARIANTS):
            c = g.generate(key, variant=v)
            p = os.path.join(d, "grass_%02d.wav" % v)
            c.to_wav(p)
            w = onset_window(c.samples, sr)
            rows.append({"cfg": name, "v": v, "path": p, "crest": crest_db(w), "attack": attack_ms(w, sr)})
            clips.append(_resample(c.samples, sr, 48000))
        print("generated %-26s %s" % (name, text))
    del g
    gc.collect()
    torch.cuda.empty_cache()

    sc = clap_scores({"all": clips}, {"base": CATEGORIES, "hard": HARD,
                                      "mat": ["the sound of a footstep on %s" % m for m in MATS],
                                      "rustle": RUSTLE}, "cuda", 32)["all"]
    for i, r in enumerate(rows):
        r["foot"] = int(sc["base"][i].argmax() == CATEGORIES.index("a footstep"))
        r["hard"] = int(sc["hard"][i].argmax() == HARD.index("a footstep"))
        r["grass"] = int(MATS[int(sc["mat"][i].argmax())] == "grass")
        r["bush"] = int(RUSTLE[int(sc["rustle"][i].argmax())] in ("walking through bushes", "rustling leaves"))

    from audiobox_aesthetics.infer import initialize_predictor
    pred = initialize_predictor()
    for i in range(0, len(rows), 40):
        chunk = rows[i: i + 40]
        for r, o in zip(chunk, pred.forward([{"path": r["path"]} for r in chunk])):
            r["PQ"] = float(o["PQ"])

    ref = {r["v"]: r for r in rows if r["cfg"] == "grassy_on_grass"}
    print()
    print(" GRASS FOOTSTEP WORDING (n=%d each, paired vs the shipped 'grassy footstep on grass')" % VARIANTS)
    print("%-26s %16s %16s %16s %16s %7s %7s %6s" % ("wording", "bush/rustle", "material grass", "clap footstep",
                                                    "clap hard", "crest", "attack", "PQ"))
    out = {}
    for name in CANDIDATES:
        rs = {r["v"]: r for r in rows if r["cfg"] == name}
        cols = []
        res = {}
        for f in ("bush", "grass", "foot", "hard"):
            a = np.array([rs[v][f] for v in ref])
            b = np.array([ref[v][f] for v in ref])
            gn, ls = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
            res[f] = {"rate": float(a.mean()), "gain": gn, "loss": ls, "p": mcnemar(gn, ls)}
            cols.append("%3.0f%% +%d/-%d p%.2f" % (100 * a.mean(), gn, ls, mcnemar(gn, ls)))
        res["crest"] = float(np.mean([r["crest"] for r in rs.values()]))
        res["attack"] = float(np.median([r["attack"] for r in rs.values()]))
        res["PQ"] = float(np.mean([r["PQ"] for r in rs.values()]))
        out[name] = res
        print("%-26s %16s %16s %16s %16s %7.1f %7.2f %6.2f" % (name, cols[0], cols[1], cols[2], cols[3],
                                                            res["crest"], res["attack"], res["PQ"]))

    seq, order = [], []
    for name in CANDIDATES:
        for v in (0, 1, 2):
            s_, a = wavfile.read(os.path.join("data", "_gate", "grass", name, "grass_%02d.wav" % v))
            a = a.astype(np.float32)
            if np.abs(a).max() > 1.5:
                a = a / 32768.0
            seq += [a, np.zeros(int(0.3 * sr), np.float32)]
        seq.append(np.zeros(int(1.2 * sr), np.float32))
        order.append(name)
    x = np.concatenate(seq)
    x = x / max(1e-9, np.abs(x).max()) * 0.9
    os.makedirs(os.path.join("data", "_gate", "listen"), exist_ok=True)
    lp = os.path.join("data", "_gate", "listen", "grass_wordings.wav")
    wavfile.write(lp, sr, (x * 32767).astype(np.int16))
    print("listening file %s, 3 takes per wording in order: %s" % (lp, " | ".join(order)))
    json.dump({"candidates": CANDIDATES, "result": out, "rows": rows},
              open(os.path.join("scripts", "bench", "out_grass_probe.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

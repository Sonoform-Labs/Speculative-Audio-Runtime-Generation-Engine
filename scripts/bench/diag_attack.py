import _bootstrap

import glob
import json
import os

import numpy as np

from sarge.loudness import normalize_loudness, target_for
from sarge.types import AudioClip
from validate_panel import CATEGORIES, _resample
from gate_quality import crest_db, attack_ms, onset_window
from ab_clap import clap_scores, mcnemar
from score_hard import HARD
from diag_room import SR

MATS = ["wood", "stone", "grass", "metal"]


def soften(x, ramp_ms):
    if ramp_ms <= 0:
        return x
    env = np.abs(x)
    t0 = int(np.argmax(env > 0.1 * env.max()))
    n = int(ramp_ms * SR / 1000)
    g = np.ones_like(x)
    seg = min(n, len(x) - t0)
    g[t0: t0 + seg] = 0.5 - 0.5 * np.cos(np.pi * np.arange(seg) / n)
    g[:t0] = 0.0
    return normalize_loudness(x * g, target_rms=target_for("normal"))


def main():
    paths = sorted(glob.glob(os.path.join("data", "_gate", "steps3_b1_pad1", "*_[0-9][0-9].wav")))
    gen = [AudioClip.from_wav(p).samples for p in paths]
    labels = [os.path.basename(p).split("_")[0] for p in paths]
    ramps = [0, 4, 8, 12, 16, 24]
    flat, tag = [], []
    for r in ramps:
        for x in gen:
            flat.append(_resample(np.asarray(soften(x, r), dtype=np.float32), SR, 48000))
            tag.append(r)
    sc = clap_scores({"all": flat}, {"hard": HARD, "base": CATEGORIES,
                                     "mat": ["the sound of a footstep on %s" % m for m in MATS]}, "cuda", 32)["all"]
    hard = sc["hard"].argmax(1) == HARD.index("a footstep")
    base = sc["base"].argmax(1) == CATEGORIES.index("a footstep")
    true = np.array([MATS.index(l) for l in labels] * len(ramps))
    mat = sc["mat"].argmax(1) == true
    ref = np.array([t == 0 for t in tag])
    out = {}
    print()
    print(" real matched crops: hard 88%%, base 95%%, attack 12.78 ms, crest 19.9 dB")
    print("%-9s %8s %8s %8s %8s %7s   %s" % ("ramp_ms", "hardFt", "baseFt", "mat", "attack", "crest", "hard vs no ramp"))
    for r in ramps:
        m = np.array([t == r for t in tag])
        xs = [soften(x, r) for x in gen]
        att = float(np.median([attack_ms(onset_window(np.asarray(x), SR), SR) for x in xs]))
        cr = float(np.mean([crest_db(onset_window(np.asarray(x), SR)) for x in xs]))
        a, b = hard[m], hard[ref]
        g, l = int((a & ~b).sum()), int((~a & b).sum())
        out[r] = {"hard": float(a.mean()), "base": float(base[m].mean()), "mat": float(mat[m].mean()),
                  "attack": att, "crest": cr, "gain": g, "loss": l, "p": mcnemar(g, l)}
        print("%-9d %7.0f%% %7.0f%% %7.0f%% %8.2f %7.1f   +%d/-%d p=%.4f" % (
            r, 100 * a.mean(), 100 * base[m].mean(), 100 * mat[m].mean(), att, cr, g, l, out[r]["p"]))
    json.dump(out, open(os.path.join("scripts", "bench", "out_diag_attack.json"), "w"), indent=1)
    print("saved -> scripts\\bench\\out_diag_attack.json")


if __name__ == "__main__":
    main()

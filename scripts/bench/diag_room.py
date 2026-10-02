import _bootstrap

import glob
import json
import os

import numpy as np
import torch
from scipy.signal import fftconvolve

from sarge.generators.stable_audio import _extract
from sarge.loudness import normalize_loudness, target_for
from sarge.types import AudioClip
from validate_panel import CATEGORIES, _resample
from gate_quality import crest_db, attack_ms, onset_window, load_wav
from ab_clap import clap_scores
from score_hard import HARD

SR = 44100


def room_ir(rt60, seed, sr=SR):
    n = int(rt60 * sr)
    t = np.arange(n) / sr
    rng = np.random.default_rng(seed)
    ir = rng.standard_normal(n) * np.exp(-6.91 * t / rt60)
    ir[0] = 0.0
    return ir / np.sqrt(np.sum(ir ** 2))


def wet(x, rt60, mix, seed):
    y = fftconvolve(x, room_ir(rt60, seed))[: len(x)]
    y = y * (np.sqrt(np.mean(x ** 2)) / (np.sqrt(np.mean(y ** 2)) + 1e-12))
    return (1 - mix) * x + mix * y


def floor(x, db, seed):
    rng = np.random.default_rng(seed + 7)
    r = np.sqrt(np.mean(x ** 2))
    return x + rng.standard_normal(len(x)) * r * 10 ** (db / 20)


def main():
    gen = [AudioClip.from_wav(p).samples for p in sorted(glob.glob(os.path.join("data", "_gate", "steps3_b1_pad1", "*_[0-9][0-9].wav")))]
    real = []
    for p in sorted(glob.glob(os.path.join("data", "real_footsteps", "*.wav"))):
        x, r = load_wav(p)
        x = _resample(x, r, SR)
        real.append(normalize_loudness(_extract(x, SR, 0.6), target_rms=target_for("normal")).astype(np.float32))
    tgt = target_for("normal")
    sets = {"real_matched": real, "gen_dry": gen}
    for rt, mix in ((0.3, 0.25), (0.3, 0.5), (0.6, 0.35)):
        sets["gen_room%.1f_mix%.2f" % (rt, mix)] = [normalize_loudness(wet(x, rt, mix, i), target_rms=tgt) for i, x in enumerate(gen)]
    sets["gen_floor-40dB"] = [normalize_loudness(floor(x, -40, i), target_rms=tgt) for i, x in enumerate(gen)]
    sets["gen_room0.3_mix0.25_floor-40dB"] = [normalize_loudness(floor(wet(x, 0.3, 0.25, i), -40, i), target_rms=tgt)
                                             for i, x in enumerate(gen)]
    sets["real_room0.3_mix0.25"] = [normalize_loudness(wet(x, 0.3, 0.25, i), target_rms=tgt) for i, x in enumerate(real)]

    flat, idx = [], []
    for name, xs in sets.items():
        for x in xs:
            flat.append(_resample(np.asarray(x, dtype=np.float32), SR, 48000))
            idx.append(name)
    sc = clap_scores({"all": flat}, {"hard": HARD, "base": CATEGORIES}, "cuda", 32)["all"]
    hard = sc["hard"].argmax(1) == HARD.index("a footstep")
    base = sc["base"].argmax(1) == CATEGORIES.index("a footstep")
    out = {}
    print()
    print("%-34s %4s %8s %8s %8s %7s" % ("set", "n", "hardFt", "baseFt", "attack", "crest"))
    for name, xs in sets.items():
        m = np.array([i == name for i in idx])
        att = [attack_ms(onset_window(np.asarray(x), SR), SR) for x in xs]
        cr = [crest_db(onset_window(np.asarray(x), SR)) for x in xs]
        out[name] = {"n": len(xs), "hard": float(hard[m].mean()), "base": float(base[m].mean()),
                     "attack": float(np.median(att)), "crest": float(np.mean(cr))}
        print("%-34s %4d %7.0f%% %7.0f%% %8.2f %7.1f" % (name, len(xs), 100 * out[name]["hard"], 100 * out[name]["base"],
                                                        out[name]["attack"], out[name]["crest"]))
    json.dump(out, open(os.path.join("scripts", "bench", "out_diag_room.json"), "w"), indent=1)
    print("saved -> scripts\\bench\\out_diag_room.json")


if __name__ == "__main__":
    main()

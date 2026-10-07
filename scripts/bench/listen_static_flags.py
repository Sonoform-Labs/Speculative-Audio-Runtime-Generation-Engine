import _bootstrap

import json
import os

import numpy as np
from scipy.io import wavfile

from sarge.generators.stable_audio3 import StableAudio3Generator
from sarge.static_guard import is_static
from sarge.types import CacheKey


def main():
    rows = json.load(open(os.path.join("scripts", "bench", "out_static_guard.json")))["rows"]
    flagged = [r for r in rows if r["first_static"]]
    g = StableAudio3Generator()
    g.ensure_ready()
    sr = int(g.sample_rate)
    seq, order = [], []
    for r in flagged:
        ev, mat, inten, ctx = r["key"].split("|")
        key = CacheKey(ev, mat, inten, ctx)
        g.static_guard = False
        raw = g.generate(key, variant=r["v"]).samples
        g.static_guard = True
        served = g.generate(key, variant=r["v"]).samples
        print("%-34s v%02d  first attempt static=%s  served static=%s" % (
            r["key"], r["v"], is_static(raw, sr)[0], is_static(served, sr)[0]))
        seq += [raw, np.zeros(int(0.4 * sr), np.float32), served, np.zeros(int(1.0 * sr), np.float32)]
        order.append("%s v%02d" % (r["key"].split("|default")[0], r["v"]))
    x = np.concatenate(seq)
    x = x / max(1e-9, np.abs(x).max()) * 0.9
    out = os.path.join("data", "_gate", "listen", "static_guard_catches.wav")
    wavfile.write(out, sr, (x * 32767).astype(np.int16))
    print("listening file %s, pairs (rejected first attempt, then served retry): %s" % (out, " | ".join(order)))


if __name__ == "__main__":
    main()

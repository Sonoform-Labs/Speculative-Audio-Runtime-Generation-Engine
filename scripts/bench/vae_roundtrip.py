import os
import sys

BENCH = os.path.join(os.getcwd(), "scripts", "bench")
if not os.path.isfile(os.path.join(BENCH, "_bootstrap.py")):
    BENCH = r"D:\Python\SAGE\scripts\bench"
sys.path.insert(0, BENCH)

import _bootstrap

import argparse
import json
import time
from collections import defaultdict

import numpy as np

import vocab_eval as ve
from sarge.generators.stable_audio3 import StableAudio3Generator

ROOT = os.path.dirname(os.path.dirname(BENCH))
COLS = ["attack_ms", "tail_ms", "floor_db", "crest_db", "low500", "decay100_db"]


def parse(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join("data", "refs_heldout", "manifest.json"))
    ap.add_argument("--win", default="0.6")
    ap.add_argument("--types", nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--out", default=os.path.join(ROOT, "scripts", "bench", "out_vae_roundtrip.json"))
    return ap.parse_args(argv)


def roundtrip(g, x):
    import torch
    sam = g._model
    n = int(sam._adapt_sample_size([{"prompt": "", "seconds_total": g.gen_duration_s}], 5292032,
                                   g.duration_padding_s))
    with torch.inference_mode():
        z, _ = sam._encode_audio_input((int(g.sample_rate), np.ascontiguousarray(x, dtype=np.float32)), n)
        y = sam.model.pretransform.decode(z)
    y = g._numpy(y)
    while y.ndim > 2:
        y = y[0]
    y = y.mean(axis=0) if y.ndim == 2 else y
    return np.ascontiguousarray(y[: len(x)], dtype=np.float32), int(z.shape[-1]), n


def med(rows, k):
    v = np.array([r[k] for r in rows], dtype=float)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else float("nan")


def main(argv=None):
    args = parse(argv)
    man = json.load(open(os.path.join(ROOT, args.manifest), encoding="utf-8"))
    by = defaultdict(list)
    for c in man["clips"]:
        t = man["sets"][c["set"]]["type"]
        if t and args.win in c["files"] and (args.types is None or t in args.types):
            by[t].append(c)
    g = StableAudio3Generator(sampler_type="euler", steps=3)
    g.ensure_ready()
    ds = int(g._model.model.pretransform.downsampling_ratio)
    out, t0 = {"manifest": args.manifest, "win": args.win, "downsampling_ratio": ds,
               "frame_ms": 1000.0 * ds / ve.SR, "types": {}}, time.perf_counter()
    for t in sorted(by):
        cs = by[t][: args.limit]
        xs = [ve.wav(os.path.join(ROOT, *c["files"][args.win].split("/"))) for c in cs]
        ys, frames = [], 0
        for x in xs:
            y, frames, n = roundtrip(g, x)
            ys.append(y)
        ro, ry = ve.shape_rows(xs), ve.shape_rows(ys)
        d_att = [b["attack_ms"] - a["attack_ms"] for a, b in zip(ro, ry)]
        err = [float(np.sqrt(np.mean((y - x) ** 2)) / max(np.sqrt(np.mean(x ** 2)), 1e-9)) for x, y in zip(xs, ys)]
        out["types"][t] = {"n": len(xs), "real": {k: med(ro, k) for k in COLS}, "roundtrip": {k: med(ry, k) for k in COLS},
                           "attack_delta_median": float(np.median(d_att)), "rel_rms_error_median": float(np.median(err)),
                           "latent_frames": frames, "audio_samples": n}
    print("VAE ROUND TRIP  small-sfx pretransform encode then decode, %s s reference windows, latent frame %.1f ms "
          "(%d samples)  %.1f s" % (args.win, out["frame_ms"], ds, time.perf_counter() - t0))
    print("  %-16s %4s  %s   attack delta  rel err" % ("type", "n", "  ".join("%-17s" % k for k in COLS)))
    for t, r in out["types"].items():
        print("  %-16s %4d  %s   %+8.2f ms  %6.3f" % (t, r["n"], "  ".join(
            "%7.3g -> %-7.3g" % (r["real"][k], r["roundtrip"][k]) for k in COLS), r["attack_delta_median"],
            r["rel_rms_error_median"]))
    json.dump(out, open(args.out, "w", encoding="utf-8"), indent=1)
    print("saved -> %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

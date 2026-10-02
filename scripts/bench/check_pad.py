import _bootstrap

import argparse
import gc
import json
import os
import time

import numpy as np
import torch

from sarge.generators.stable_audio3 import StableAudio3Generator
from sarge.types import CacheKey
from validate_panel import _resample
from gate_quality import crest_db, attack_ms, onset_window
from ab_clap import clap_scores, mcnemar

KEYS = [CacheKey("impact", "glass", "normal"), CacheKey("impact", "metal", "normal"),
        CacheKey("impact", "wood", "normal"), CacheKey("impact", "stone", "normal"),
        CacheKey("splash", "water", "normal"), CacheKey("land", "grass", "heavy"),
        CacheKey("jump", "stone", "normal"), CacheKey("footstep", "gravel", "normal")]
CONFIGS = {
    "pad6_s3": dict(duration_padding_s=6.0, steps=3, mid_noise=None),
    "pad1_s3": dict(duration_padding_s=1.0, steps=3, mid_noise=None),
    "pad1_s2_m68": dict(duration_padding_s=1.0, steps=2, mid_noise=0.68),
}


def label(k):
    return "%s on %s" % (k.event_type, k.material)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", type=int, default=10)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default=os.path.join("scripts", "bench", "out_check_pad.json"))
    args = ap.parse_args()

    g = StableAudio3Generator()
    g.ensure_ready()
    sr = int(g.sample_rate)
    clips, rows, timing = [], [], {}
    for name, cfg in CONFIGS.items():
        for a, v in cfg.items():
            setattr(g, a, v)
        d = os.path.join("data", "_gate", "padcheck", name)
        os.makedirs(d, exist_ok=True)
        k0 = KEYS[0]
        g.generate(k0, variant=0)
        ts = []
        for _ in range(args.reps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            g.generate(k0, variant=0)
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000.0)
        timing[name] = float(np.median(ts))
        for k in KEYS:
            for v in range(args.variants):
                c = g.generate(k, variant=v)
                c.to_wav(os.path.join(d, "%s-%s_%02d.wav" % (k.event_type, k.material, v)))
                clips.append(c.samples)
                w = onset_window(c.samples, sr)
                rows.append({"cfg": name, "key": label(k), "v": v, "crest": crest_db(w), "attack": attack_ms(w, sr),
                             "path": os.path.join(d, "%s-%s_%02d.wav" % (k.event_type, k.material, v))})
        print("generated %-12s median %.1f ms (impact glass, production fast path)  %s"
              % (name, timing[name], g.fast_stats()))
    del g
    gc.collect()
    torch.cuda.empty_cache()

    labels = [label(k) for k in KEYS]
    sc = clap_scores({"all": [_resample(x, sr, 48000) for x in clips]}, {"id": ["the sound of a %s" % t for t in labels]},
                     "cuda", 32)["all"]["id"]
    pred = sc.argmax(1)
    for i, r in enumerate(rows):
        r["hit"] = int(labels[int(pred[i])] == r["key"])

    from audiobox_aesthetics.infer import initialize_predictor
    pred_ab = initialize_predictor()
    for i in range(0, len(rows), 40):
        chunk = rows[i: i + 40]
        for r, o in zip(chunk, pred_ab.forward([{"path": r["path"]} for r in chunk])):
            r["PQ"] = float(o["PQ"])

    ref = {(r["key"], r["v"]): r for r in rows if r["cfg"] == "pad6_s3"}
    print()
    print("%-12s %7s %6s %7s %7s %6s   %-22s %s" % ("config", "ms", "id", "crest", "attack", "PQ", "id vs pad6_s3", "per key hits"))
    print("-" * 120)
    summary = {}
    for name in CONFIGS:
        rs = {(r["key"], r["v"]): r for r in rows if r["cfg"] == name}
        a = np.array([rs[k]["hit"] for k in ref])
        b = np.array([ref[k]["hit"] for k in ref])
        gn, ls = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
        per = " ".join("%d" % sum(rs[(lb, v)]["hit"] for v in range(args.variants)) for lb in labels)
        s = {"ms": timing[name], "id": float(a.mean()), "gain": gn, "loss": ls, "p": mcnemar(gn, ls),
             "crest": float(np.mean([r["crest"] for r in rs.values()])),
             "attack": float(np.median([r["attack"] for r in rs.values()])),
             "PQ": float(np.mean([r["PQ"] for r in rs.values()])),
             "d_PQ": float(np.mean([rs[k]["PQ"] - ref[k]["PQ"] for k in ref]))}
        summary[name] = s
        print("%-12s %7.1f %5.0f%% %7.1f %7.2f %6.2f   +%d/-%d p=%.3f dPQ %+.2f   %s" % (
            name, s["ms"], 100 * s["id"], s["crest"], s["attack"], s["PQ"], gn, ls, s["p"], s["d_PQ"], per))
    print(" keys: " + " | ".join(labels))
    json.dump({"summary": summary, "rows": rows}, open(args.out, "w"), indent=1)
    print("saved -> %s" % args.out)


if __name__ == "__main__":
    main()

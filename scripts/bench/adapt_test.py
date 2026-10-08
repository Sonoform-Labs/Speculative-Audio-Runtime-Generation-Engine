import _bootstrap

import argparse
import gc
import json
import os
import subprocess
import sys
import time

import numpy as np
from scipy.io import wavfile

from sarge.sidecar.client import connect
from sarge.static_guard import is_static

NOVEL = [
    ("shoot", "rifle", "a rifle gunshot"),
    ("shoot", "pistol", "a pistol gunshot"),
    ("reload", "rifle", "a gun being reloaded"),
    ("equip", "rifle", "a weapon being drawn"),
    ("empty_click", "pistol", "an empty gun clicking"),
    ("explosion", "grenade", "a grenade explosion"),
    ("swing", "sword", "a sword swinging through the air"),
    ("open", "door", "a door opening"),
    ("slam", "door", "a door slamming shut"),
    ("break", "window", "a window breaking"),
    ("click", "button", "a user interface click"),
    ("cast", "fire_spell", "a magic fire spell"),
    ("punch", "body", "a punch hitting a body"),
    ("horn", "truck", "a truck horn"),
    ("bark", "dog", "a dog barking"),
]
SR = 44100


def ev(c, et, mat, log, t0, phase):
    s = time.perf_counter()
    body, x = c.event(et, mat)
    ms = (time.perf_counter() - s) * 1000.0
    log.append({"phase": phase, "t": time.perf_counter() - t0, "key": "%s|%s" % (et, mat), "ms": ms,
                "served": body.get("served"), "variant": body.get("variant_id"), "intent": body.get("intent", ""),
                "learned": body.get("learned"), "static": bool(is_static(np.asarray(x, dtype=np.float32), SR)[0]) if len(x) else True})
    return np.asarray(x, dtype=np.float32)


def session(c, log, t0):
    c.context(time.perf_counter() - t0, "rifle", ["concrete"], "range")
    ev(c, "equip", "rifle", log, t0, "session")
    for cycle in range(3):
        for burst in range(3):
            for _ in range(5):
                ev(c, "shoot", "rifle", log, t0, "session")
                time.sleep(0.12)
            time.sleep(0.9)
        ev(c, "reload", "rifle", log, t0, "session")
        time.sleep(1.5)
    for _ in range(12):
        ev(c, "footstep", "concrete", log, t0, "session")
        time.sleep(0.45)
    c.context(time.perf_counter() - t0, "pistol", ["concrete"], "range")
    ev(c, "equip", "pistol", log, t0, "session")
    for cycle in range(3):
        for _ in range(6):
            ev(c, "shoot", "pistol", log, t0, "session")
            time.sleep(0.4)
        ev(c, "empty_click", "pistol", log, t0, "session")
        time.sleep(0.3)
        ev(c, "reload", "pistol", log, t0, "session")
        time.sleep(1.5)
    c.context(time.perf_counter() - t0, "rifle", ["concrete"], "range")
    ev(c, "equip", "rifle", log, t0, "session")
    for _ in range(10):
        ev(c, "shoot", "rifle", log, t0, "session")
        time.sleep(0.12)


def summarize(log):
    s = [r for r in log if r["phase"] == "session"]
    out = {}
    for key in sorted({r["key"] for r in s}):
        rs = [r for r in s if r["key"] == key]
        served = [r["served"] for r in rs]
        out[key] = {"n": len(rs), "realtime": served.count("realtime"), "cached": served.count("cached"),
                    "other": len(rs) - served.count("realtime") - served.count("cached"),
                    "first_ms": rs[0]["ms"], "first_served": rs[0]["served"],
                    "p50_ms": float(np.median([r["ms"] for r in rs])), "max_ms": float(max(r["ms"] for r in rs)),
                    "distinct_variants": len({r["variant"] for r in rs}),
                    "repeat_in_a_row": sum(1 for a, b in zip(rs, rs[1:]) if a["variant"] == b["variant"]),
                    "static_served": sum(r["static"] for r in rs)}
    return out


def score(clips, labels, texts):
    import torch
    from validate_panel import _resample
    from ab_clap import clap_scores
    sc = clap_scores({"all": [_resample(x, SR, 48000) for x in clips]}, {"t": texts}, "cuda", 32)["all"]["t"]
    top = sc.argmax(1)
    ranks = []
    for i, lab in enumerate(labels):
        order = list(np.argsort(-sc[i]))
        ranks.append(order.index(texts.index(lab)) + 1)
    return [texts[int(t)] for t in top], ranks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8799)
    ap.add_argument("--out", default=os.path.join("scripts", "bench", "out_adapt_test.json"))
    args = ap.parse_args()

    srv_log = open(os.path.join("scripts", "bench", "adapt_sidecar.log"), "w")
    srv = subprocess.Popen([sys.executable, "-u", os.path.join("scripts", "run_sidecar.py"), "--generator", "sa3",
                            "--tier", "default", "--realtime", "--learn", "--port", str(args.port)],
                           stdout=srv_log, stderr=subprocess.STDOUT)
    log, novel_clips = [], []
    try:
        c = None
        deadline = time.time() + 240
        while time.time() < deadline and c is None:
            try:
                c = connect(port=args.port, timeout=30.0)
            except OSError:
                time.sleep(1.0)
        if c is None:
            raise SystemExit("sidecar never came up")
        c.hello()
        t0 = time.perf_counter()
        for et, mat, _ in NOVEL:
            novel_clips.append(ev(c, et, mat, log, t0, "novel"))
        for et, mat, _ in NOVEL:
            ev(c, et, mat, log, t0, "novel_repeat")
        session(c, log, t0)
        stats = c.stats()
        c.close()
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=30)
        except Exception:
            srv.kill()
        srv_log.close()
    gc.collect()
    time.sleep(3.0)

    texts = [d for _, _, d in NOVEL]
    tops, ranks = score(novel_clips, texts, texts)
    nov = [r for r in log if r["phase"] == "novel"]
    rep = [r for r in log if r["phase"] == "novel_repeat"]
    print()
    print(" NEW VOCABULARY (never configured): first hit, repeat, and whether CLAP hears what was asked among %d labels" % len(texts))
    print("%-22s %9s %-9s %9s %-8s %5s  %s" % ("key", "first ms", "served", "repeat ms", "static", "rank", "CLAP top-1"))
    hits = 0
    for r, rr, (et, mat, d), top, rk in zip(nov, rep, NOVEL, tops, ranks):
        hits += int(rk == 1)
        print("%-22s %9.1f %-9s %9.1f %-8s %5d  %s%s" % (r["key"], r["ms"], r["served"], rr["ms"], r["static"], rk, top,
                                                       "" if top == d else "   (asked: %s)" % d))
    print(" CLAP identity: %d/%d heard as asked (chance %.0f%%); median rank %.1f of %d" % (
        hits, len(texts), 100.0 / len(texts), float(np.median(ranks)), len(texts)))

    summ = summarize(log)
    print()
    print(" GAMEPLAY SESSION (rifle bursts, reloads, walking, switch to pistol, switch back), no hints from the game")
    print("%-22s %4s %9s %7s %10s %8s %8s %9s %8s %7s" % ("key", "n", "realtime", "cached", "first", "p50 ms", "max ms",
                                                     "variants", "repeats", "static"))
    for k, v in summ.items():
        print("%-22s %4d %9d %7d %10s %8.1f %8.1f %9d %8d %7d" % (k, v["n"], v["realtime"], v["cached"],
                                                                 "%.0f/%s" % (v["first_ms"], v["first_served"][:4]),
                                                                 v["p50_ms"], v["max_ms"], v["distinct_variants"],
                                                                 v["repeat_in_a_row"], v["static_served"]))
    s = [r for r in log if r["phase"] == "session"]
    intents = [r["intent"] for r in s if r["intent"]]
    print(" session events %d, served realtime %d, cached %d; events with an intent prediction %d; last intent '%s'" % (
        len(s), sum(r["served"] == "realtime" for r in s), sum(r["served"] == "cached" for r in s), len(intents),
        intents[-1] if intents else ""))
    print(" server stats %s" % json.dumps(stats.get("stats", stats))[:400])

    seq = []
    for x in novel_clips:
        seq += [x, np.zeros(int(0.35 * SR), np.float32)]
    y = np.concatenate(seq)
    y = y / max(1e-9, np.abs(y).max()) * 0.9
    lp = os.path.join("data", "_gate", "listen", "new_vocabulary.wav")
    os.makedirs(os.path.dirname(lp), exist_ok=True)
    wavfile.write(lp, SR, (y * 32767).astype(np.int16))
    print("listening file %s in order: %s" % (lp, " | ".join("%s %s" % (e, m) for e, m, _ in NOVEL)))
    json.dump({"novel": list(zip([r["key"] for r in nov], tops, ranks)), "log": log, "session": summ, "stats": stats},
              open(args.out, "w"), indent=1, default=str)
    print("saved -> %s" % args.out)


if __name__ == "__main__":
    main()

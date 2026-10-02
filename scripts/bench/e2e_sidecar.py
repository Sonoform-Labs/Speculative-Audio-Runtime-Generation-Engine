import _bootstrap

import json
import os
import sys
import time

import numpy as np

from sarge.sidecar.client import connect

FIRST = [("footstep", "wood"), ("footstep", "stone"), ("impact", "glass"), ("impact", "metal"),
         ("footstep", "grass"), ("land", "stone"), ("footstep", "gravel"), ("jump", "wood")]


def timed_event(c, ev, mat, intensity="normal"):
    t0 = time.perf_counter()
    body, samples = c.event(ev, mat, intensity)
    return (time.perf_counter() - t0) * 1000.0, body, samples


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    deadline = time.time() + 180
    c = None
    while time.time() < deadline:
        try:
            c = connect(port=port, timeout=30.0)
            break
        except OSError:
            time.sleep(1.0)
    if c is None:
        raise SystemExit("sidecar never came up")
    hello = c.hello()
    print("welcome", {k: hello.get(k) for k in ("proto", "server", "sample_rate", "pcm")})

    res = {"first": [], "repeat": [], "predicted": None}
    for ev, mat in FIRST:
        ms, body, s = timed_event(c, ev, mat)
        res["first"].append({"key": "%s|%s" % (ev, mat), "ms": ms, "served": body.get("served", body.get("source")),
                             "n": int(len(s)), "rms": float(np.sqrt(np.mean(np.square(s)))) if len(s) else 0.0})
    for ev, mat in FIRST:
        ms, body, s = timed_event(c, ev, mat)
        res["repeat"].append({"key": "%s|%s" % (ev, mat), "ms": ms, "served": body.get("served", body.get("source"))})

    ack = c.predict([{"event_type": "footstep", "material": "metal", "intensity": "normal",
                      "eta_s": 0.4, "confidence": 0.9, "source": "motion"}], now=0.0)
    time.sleep(0.4)
    ms, body, s = timed_event(c, "footstep", "metal")
    res["predicted"] = {"ack": ack, "ms": ms, "served": body.get("served", body.get("source"))}
    res["stats"] = c.stats()
    c.close()

    print()
    print("%-18s %9s  %-10s %8s" % ("first hit", "ms", "served", "rms"))
    for r in res["first"]:
        print("%-18s %9.1f  %-10s %8.3f" % (r["key"], r["ms"], r["served"], r["rms"]))
    print()
    print("%-18s %9s  %-10s" % ("repeat", "ms", "served"))
    for r in res["repeat"]:
        print("%-18s %9.1f  %-10s" % (r["key"], r["ms"], r["served"]))
    p = res["predicted"]
    print()
    print("predicted footstep|metal 400 ms ahead: ack %s, fired after 400 ms -> %.1f ms, served %s"
          % (p["ack"], p["ms"], p["served"]))
    f = [r["ms"] for r in res["first"]]
    rp = [r["ms"] for r in res["repeat"]]
    print("first-hit median %.1f ms (max %.1f), repeat median %.1f ms" % (np.median(f), max(f), np.median(rp)))
    json.dump(res, open(os.path.join("scripts", "bench", "out_e2e_sidecar.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()

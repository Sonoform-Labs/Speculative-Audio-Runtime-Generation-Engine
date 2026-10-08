import _bootstrap

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

GAME = r'''
import json, sys, time
import torch
target_ms, seconds, out = float(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
dev = "cuda"
a = torch.randn(2048, 2048, device=dev, dtype=torch.float16)
b = torch.randn(2048, 2048, device=dev, dtype=torch.float16)
def work(k):
    c = a
    for _ in range(k):
        c = torch.mm(c, b) * 0.001
    return c
for _ in range(5):
    work(4)
torch.cuda.synchronize()
k = 1
while True:
    t0 = time.perf_counter()
    for _ in range(5):
        work(k)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 200.0
    if ms >= target_ms or k > 4096:
        break
    k = max(k + 1, int(k * target_ms / max(ms, 1e-3)))
frame = 1.0 / 60.0
times, starts = [], []
t_end = time.perf_counter() + seconds
nxt = time.perf_counter()
open(out + ".ready", "w").write("1")
while time.perf_counter() < t_end:
    t0 = time.perf_counter()
    work(k)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    starts.append(t0)
    times.append((t1 - t0) * 1000.0)
    nxt += frame
    d = nxt - time.perf_counter()
    if d > 0:
        time.sleep(d)
    else:
        nxt = time.perf_counter()
json.dump({"k": k, "target_ms": target_ms, "gpu_ms": times, "start": starts}, open(out, "w"))
'''


def game_stats(d):
    g = np.array(d["gpu_ms"])
    s = np.array(d["start"])
    gaps = np.diff(s) * 1000.0
    return {"frames": int(len(g)), "work_p50": float(np.median(g)), "work_p95": float(np.percentile(g, 95)),
            "work_p99": float(np.percentile(g, 99)), "work_max": float(g.max()),
            "frame_p50": float(np.median(gaps)), "frame_p99": float(np.percentile(gaps, 99)),
            "over_16_7": float(np.mean(gaps > 16.7 * 1.05)), "over_33": float(np.mean(gaps > 33.4))}


def run_game(target_ms, seconds, tag):
    out = os.path.join("scripts", "bench", "_game_%s.json" % tag)
    for f in (out, out + ".ready"):
        if os.path.exists(f):
            os.remove(f)
    p = subprocess.Popen([sys.executable, "-c", GAME, str(target_ms), str(seconds), out],
                         creationflags=getattr(subprocess, "NORMAL_PRIORITY_CLASS", 0))
    while not os.path.exists(out + ".ready"):
        if p.poll() is not None:
            raise SystemExit("game process died")
        time.sleep(0.05)
    return p, out


def sarge_loop(g, seconds, interval, v0):
    from sarge.types import CacheKey
    import torch
    mats = ["wood", "stone", "grass", "metal", "gravel", "dirt"]
    lat = []
    t_end = time.perf_counter() + seconds
    nxt = time.perf_counter()
    v = v0
    while time.perf_counter() < t_end:
        key = CacheKey("footstep", mats[v % len(mats)], "normal")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        g.generate(key, variant=v)
        torch.cuda.synchronize()
        lat.append((time.perf_counter() - t0) * 1000.0)
        v += 1
        nxt += interval
        d = nxt - time.perf_counter()
        if d > 0:
            time.sleep(d)
        else:
            nxt = time.perf_counter()
    a = np.array(lat)
    return {"clips": int(len(a)), "p50": float(np.median(a)), "p95": float(np.percentile(a, 95)), "max": float(a.max())}, v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--interval", type=float, default=0.4)
    ap.add_argument("--loads", default="6,12")
    ap.add_argument("--priority", default="normal", choices=["normal", "below", "idle"])
    ap.add_argument("--out", default=os.path.join("scripts", "bench", "out_gpu_contention.json"))
    args = ap.parse_args()
    if args.priority != "normal":
        import ctypes
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        cls = {"below": 0x00004000, "idle": 0x00000040}[args.priority]
        ok = k32.SetPriorityClass(k32.GetCurrentProcess(), cls)
        print("sarge process priority %s set=%s" % (args.priority, bool(ok)))

    from sarge.generators.stable_audio3 import StableAudio3Generator
    from sarge.types import CacheKey
    g = StableAudio3Generator()
    g.ensure_ready()
    for v in range(3):
        g.generate(CacheKey("footstep", "wood", "normal"), variant=1000 + v)

    res = {"interval_s": args.interval, "seconds": args.seconds, "priority": args.priority}
    alone, v = sarge_loop(g, args.seconds, args.interval, 2000)
    res["sarge_alone"] = alone
    print("sarge alone: %d clips  p50 %.1f ms  p95 %.1f  max %.1f" % (alone["clips"], alone["p50"], alone["p95"], alone["max"]))

    for load in [float(x) for x in args.loads.split(",")]:
        p, out = run_game(load, args.seconds, "alone_%g" % load)
        p.wait()
        ga = game_stats(json.load(open(out)))
        p, out = run_game(load, args.seconds + 2.0, "both_%g" % load)
        time.sleep(1.0)
        both_s, v = sarge_loop(g, args.seconds, args.interval, v)
        p.wait()
        gb = game_stats(json.load(open(out)))
        res["load_%g" % load] = {"game_alone": ga, "game_with_sarge": gb, "sarge_with_game": both_s}
        print()
        print("game load %.0f ms GPU per 60 fps frame (%.0f%% of a 16.7 ms frame)" % (load, 100 * load / 16.67))
        print("  game alone      frame work p50 %.1f p99 %.1f max %.1f ms | frame interval p99 %.1f | late frames %.1f%%  >33 ms %.1f%%" % (
            ga["work_p50"], ga["work_p99"], ga["work_max"], ga["frame_p99"], 100 * ga["over_16_7"], 100 * ga["over_33"]))
        print("  game + sarge    frame work p50 %.1f p99 %.1f max %.1f ms | frame interval p99 %.1f | late frames %.1f%%  >33 ms %.1f%%" % (
            gb["work_p50"], gb["work_p99"], gb["work_max"], gb["frame_p99"], 100 * gb["over_16_7"], 100 * gb["over_33"]))
        print("  sarge + game    %d clips  p50 %.1f ms  p95 %.1f  max %.1f  (alone p50 %.1f)" % (
            both_s["clips"], both_s["p50"], both_s["p95"], both_s["max"], alone["p50"]))
    json.dump(res, open(args.out, "w"), indent=1)
    print("saved -> %s" % args.out)


if __name__ == "__main__":
    main()
